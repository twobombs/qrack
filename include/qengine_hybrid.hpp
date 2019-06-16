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
    virtual void Finish();
    virtual bool isFinished();
    virtual bool TrySeparate(bitLenInt start, bitLenInt length = 1);

    /** @} */

    /**
     * \defgroup QInterface pure virtuals (not overriden by QEngine)
     *@{
     */
    virtual void SetQuantumState(const complex* inputState) = 0;
    virtual void GetQuantumState(complex* outputState) = 0;
    virtual void GetProbs(real1* outputProbs) = 0;
    virtual complex GetAmplitude(bitCapInt perm) = 0;
    virtual void SetPermutation(bitCapInt perm, complex phaseFac = complex(-999.0, -999.0)) = 0;

    virtual void CSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2) = 0;
    virtual void AntiCSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2) = 0;
    virtual void CSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2) = 0;
    virtual void AntiCSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2) = 0;
    virtual void CISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2) = 0;
    virtual void AntiCISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2) = 0;

    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length) = 0;
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen) = 0;
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex) = 0;
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex) = 0;
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length) = 0;
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;
    virtual void DECSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex) = 0;
    virtual void DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length) = 0;
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length) = 0;
    virtual void MULModNOut(
        bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length) = 0;
    virtual void POWModNOut(
        bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length) = 0;
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen) = 0;
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen) = 0;
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen) = 0;
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen) = 0;

    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length) = 0;
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex) = 0;
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length) = 0;
    virtual void PhaseFlip() = 0;

    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values) = 0;
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values) = 0;
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values) = 0;

    virtual void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) = 0;
    virtual void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) = 0;
    virtual void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) = 0;

    virtual void CopyState(QInterfacePtr orig) = 0;
    virtual real1 Prob(bitLenInt qubitIndex) = 0;
    virtual real1 ProbAll(bitCapInt fullRegister) = 0;
    virtual bool ApproxCompare(QInterfacePtr toCompare) = 0;
    virtual void UpdateRunningNorm() = 0;
    virtual void NormalizeState(real1 nrm = -999.0) = 0;
    virtual QInterfacePtr Clone() = 0;

    /** @} */
};
} // namespace Qrack
