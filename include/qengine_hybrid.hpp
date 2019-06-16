//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// QEngineHybrid is a header-only wrapper that selects between QEngineCPU and QEngineOCL implementations based on a
// qubit threshold for maximum performance.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "qengine_cpu.hpp"
#include "qengine_opencl.hpp"

namespace Qrack {

#define QENGINGEHYBRID_CALL(method)                                                                                    \
    if (isLocked) {                                                                                                    \
        return QEngineCPU::method;                                                                                     \
    } else {                                                                                                           \
        return QEngineOCL::method;                                                                                     \
    }

class QEngineHybrid;
typedef std::shared_ptr<QEngineHybrid> QEngineHybridPtr;

class QEngineHybrid : public QEngineCPU, QEngineOCL, QInterface {
protected:
    bitLenInt minimumOCLBits;
    bool isLocked;

    void Lock()
    {
        if (isLocked) {
            return;
        }

        LockSync(CL_MAP_READ | CL_MAP_WRITE);
        isLocked = true;
    }

    void Unlock()
    {
        if (!isLocked) {
            return;
        }

        UnlockSync();
        isLocked = false;
    }

    void SyncToOther(QInterfacePtr oth)
    {
        QEngineHybridPtr other = std::dynamic_pointer_cast<QEngineHybrid>(oth);

        if (isLocked != other->isLocked) {
            if (isLocked) {
                Unlock();
            } else {
                Lock();
            }
        }
    }

    void AdjustToLengthChange()
    {
        bool nowLocked = (minimumOCLBits >= qubitCount);

        if (isLocked != nowLocked) {
            if (isLocked) {
                Unlock();
            } else {
                Lock();
            }
        }
    }

public:
    // Only the constructor and compose/decompose methods need special implementations. All other methods in the public
    // interface just switch between QEngineCPU and QEngineOCL
    QEngineHybrid(bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = complex(-999.0, -999.0), bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int devId = -1, bool useHardwareRNG = true, bitLenInt minOCLBits = 3)
        : QEngine(qBitCount, rgp, doNorm, randomGlobalPhase, useHostMem, useHardwareRNG)
        , minimumOCLBits(minOCLBits)
        , isLocked(false)
    {
        nrmArray = NULL;
        deviceID = devId;
        unlockHostMem = false;

        InitOCL(deviceID);

        SetPermutation(initState, phaseFac);

        if (qubitCount < minOCLBits) {
            Lock();
        }
    }

    /**
     * \defgroup Externally syncing implementations
     *@{
     */

    virtual bitLenInt Compose(QInterfacePtr toCopy)
    {
        SyncToOther(toCopy);
        QENGINGEHYBRID_CALL(Compose(toCopy));
        AdjustToLengthChange();
    }
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        SyncToOther(toCopy);
        QENGINGEHYBRID_CALL(Compose(toCopy, start));
        AdjustToLengthChange();
    }

    virtual void Decompose(bitLenInt start, bitLenInt length, QInterfacePtr dest)
    {
        SyncToOther(dest);
        QENGINGEHYBRID_CALL(Decompose(start, length, dest));
        AdjustToLengthChange();
    }

    virtual void Dispose(bitLenInt start, bitLenInt length)
    {
        QENGINGEHYBRID_CALL(Dispose(start, length));
        AdjustToLengthChange();
    }

    virtual bool TryDecompose(bitLenInt start, bitLenInt length, QInterfacePtr dest)
    {
        SyncToOther(dest);
        QENGINGEHYBRID_CALL(TryDecompose(start, length, dest));
        AdjustToLengthChange();
    }

    virtual bool ApproxCompare(QInterfacePtr toCompare)
    {
        SyncToOther(toCompare);
        QENGINGEHYBRID_CALL(ApproxCompare(toCompare));
        AdjustToLengthChange();
    }

    virtual void CopyState(QInterfacePtr orig)
    {
        SyncToOther(orig);
        QENGINGEHYBRID_CALL(CopyState(orig));
        AdjustToLengthChange();
    }

    /** @} */

    /**
     * \defgroup QEngine overrides
     *@{
     */

    virtual void INCDECC(
        bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex)
    {
        QENGINGEHYBRID_CALL(INCDECC(toMod, inOutStart, length, carryIndex));
    }
    virtual void INCDECSC(bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length,
        const bitLenInt& overflowIndex, const bitLenInt& carryIndex)
    {
        QENGINGEHYBRID_CALL(INCDECSC(toMod, inOutStart, length, overflowIndex, carryIndex));
    }
    virtual void INCDECSC(
        bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex)
    {
        QENGINGEHYBRID_CALL(INCDECSC(toMod, inOutStart, length, carryIndex));
    }
    virtual void INCDECBCDC(
        bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex)
    {
        QENGINGEHYBRID_CALL(INCDECBCDC(toMod, inOutStart, length, carryIndex));
    }

    /** @} */

    /**
     * \defgroup QEngineOCL overrides
     *@{
     */
    virtual void Finish() { QENGINGEHYBRID_CALL(Finish()); }
    virtual bool isFinished() { QENGINGEHYBRID_CALL(isFinished()); }
    virtual bool TrySeparate(bitLenInt start, bitLenInt length = 1) { QENGINGEHYBRID_CALL(TrySeparate(start, length)); }

    /** @} */

    /**
     * \defgroup QInterface pure virtuals (not overriden by QEngine)
     *@{
     */
    virtual void SetQuantumState(const complex* inputState) { QENGINGEHYBRID_CALL(SetQuantumState(inputState)); }
    virtual void GetQuantumState(complex* outputState) { QENGINGEHYBRID_CALL(GetQuantumState(outputState)); }
    virtual void GetProbs(real1* outputProbs) { QENGINGEHYBRID_CALL(GetProbs(outputProbs)); }
    virtual complex GetAmplitude(bitCapInt perm) { QENGINGEHYBRID_CALL(GetAmplitude(perm)); }
    virtual void SetPermutation(bitCapInt perm, complex phaseFac = complex(-999.0, -999.0))
    {
        QENGINGEHYBRID_CALL(SetPermutation(perm, phaseFac));
    }

    virtual void CSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        QENGINGEHYBRID_CALL(CSwap(controls, controlLen, qubit1, qubit2));
    }
    virtual void AntiCSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        QENGINGEHYBRID_CALL(AntiCSwap(controls, controlLen, qubit1, qubit2));
    }
    virtual void CSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        QENGINGEHYBRID_CALL(CSqrtSwap(controls, controlLen, qubit1, qubit2));
    }
    virtual void AntiCSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        QENGINGEHYBRID_CALL(AntiCSqrtSwap(controls, controlLen, qubit1, qubit2));
    }
    virtual void CISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        QENGINGEHYBRID_CALL(CISqrtSwap(controls, controlLen, qubit1, qubit2));
    }
    virtual void AntiCISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        QENGINGEHYBRID_CALL(AntiCISqrtSwap(controls, controlLen, qubit1, qubit2));
    }

    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        QENGINGEHYBRID_CALL(INC(toAdd, start, length));
    }
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
    {
        QENGINGEHYBRID_CALL(CINC(toAdd, inOutStart, length, controls, controlLen));
    }
    virtual void INCS(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex)
    {
        QENGINGEHYBRID_CALL(INCS(toAdd, inOutStart, length, overflowIndex));
    }
    virtual void INCBCD(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length)
    {
        QENGINGEHYBRID_CALL(INCBCD(toAdd, inOutStart, length));
    }
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        QENGINGEHYBRID_CALL(MUL(toMul, inOutStart, carryStart, length));
    }
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        QENGINGEHYBRID_CALL(MUL(toDiv, inOutStart, carryStart, length));
    }
    virtual void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        QENGINGEHYBRID_CALL(MULModNOut(toMul, modN, inOutStart, outStart, length));
    }
    virtual void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        QENGINGEHYBRID_CALL(POWModNOut(base, modN, inOutStart, outStart, length));
    }
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        QENGINGEHYBRID_CALL(CMUL(toMul, inOutStart, carryStart, length, controls, controlLen));
    }
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        QENGINGEHYBRID_CALL(CDIV(toDiv, inOutStart, carryStart, length, controls, controlLen));
    }
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        QENGINGEHYBRID_CALL(CMULModNOut(toMul, modN, inOutStart, outStart, length, controls, controlLen));
    }
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        QENGINGEHYBRID_CALL(CPOWModNOut(base, modN, inOutStart, outStart, length, controls, controlLen));
    }

    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length) { QENGINGEHYBRID_CALL(ZeroPhaseFlip(start, length)); }
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        QENGINGEHYBRID_CALL(CPhaseFlipIfLess(greaterPerm, start, length, flagIndex));
    }
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
    {
        QENGINGEHYBRID_CALL(PhaseFlipIfLess(greaterPerm, start, length));
    }
    virtual void PhaseFlip() { QENGINGEHYBRID_CALL(PhaseFlip()); }

    virtual bitCapInt IndexedLDA(
        bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, unsigned char* values)
    {
        QENGINGEHYBRID_CALL(IndexedLDA(indexStart, indexLength, valueStart, valueLength, values));
    }
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
    {
        QENGINGEHYBRID_CALL(IndexedADC(indexStart, indexLength, valueStart, valueLength, carryIndex, values));
    }
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
    {
        QENGINGEHYBRID_CALL(IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values));
    }

    virtual void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        QENGINGEHYBRID_CALL(Swap(qubitIndex1, qubitIndex2));
    }
    virtual void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        QENGINGEHYBRID_CALL(SqrtSwap(qubitIndex1, qubitIndex2));
    }
    virtual void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        QENGINGEHYBRID_CALL(ISqrtSwap(qubitIndex1, qubitIndex2));
    }

    virtual real1 Prob(bitLenInt qubitIndex) { QENGINGEHYBRID_CALL(Prob(qubitIndex)); }
    virtual real1 ProbAll(bitCapInt fullRegister) { QENGINGEHYBRID_CALL(ProbAll(fullRegister)); }
    virtual void UpdateRunningNorm() { QENGINGEHYBRID_CALL(UpdateRunningNorm()); }
    virtual void NormalizeState(real1 nrm = -999.0) { QENGINGEHYBRID_CALL(NormalizeState(nrm)); }
    virtual QInterfacePtr Clone() { QENGINGEHYBRID_CALL(Clone()); }

    /** @} */
};
} // namespace Qrack
