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
};
} // namespace Qrack
