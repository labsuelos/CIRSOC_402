'''Module for testing the Load and Load classes in cirsoc_402.loads
'''

import copy
import numpy as np
import pytest
import itertools

from cirsoc_402.load import ReferenceFrame
from cirsoc_402.load.loadclass import Load
from cirsoc_402.load.loadclass import GenericLoad
from cirsoc_402.load.loadgroup import LoadGroup
from cirsoc_402.load import Load
from cirsoc_402.constants import LOAD


def test_init():
    '''Test __init__ method from LoadGroup class
    '''

    # empty
    group = LoadGroup()
    for loadid in LOAD:
        load = getattr(group, loadid)
        assert load == Load(loadid)
    

def test_repr():
    '''Test __repr__ method from LoadGroup class
    '''
    group = LoadGroup()
    for loadid in LOAD:
        group = group + Load.fromarray(loadid, force=np.random.random(3),
                                       moment=np.random.random(3))
    
    row = "{}\t{:.2f}\t\t{:.2f}\t\t{:.2f}\t\t\t{:.2f}\t\t{:.2f}\t\t{:.2f}\n"

    txt = "Load\tFx\t\tFy\t\tFz\t\t\tMx\t\tMy\t\tMz\n"
    for loadid in LOAD:
        load = getattr(group, loadid)
        force = load.force
        moment = load.moment
        txt += row.format(loadid, force[0], force[1], force[2],
                            moment[0], moment[1], moment[2])
    txt += "\n"
    refframe = group.D.reference
    txt += "At reference frame:\n"
    txt += "R = ({:.2f}, {:.2f}, {:.2f})\n".format(refframe.origin[0], refframe.origin[1], refframe.origin[2])
    txt += "ex = ({:.2f}, {:.2f}, {:.2f})\n".format(refframe.xversor[0], refframe.xversor[1], refframe.xversor[2])
    txt += "ey = ({:.2f}, {:.2f}, {:.2f})\n".format(refframe.yversor[0], refframe.yversor[1], refframe.yversor[2])
    txt += "ez = ({:.2f}, {:.2f}, {:.2f})".format(refframe.zversor[0], refframe.zversor[1], refframe.zversor[2])

    assert group.__repr__() == txt


def test_shift():
    '''Test shift method from LoadGroup class
    '''

    group = LoadGroup()
    loads = {}
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                       moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    
    shifts = [-23.5, -0.6, 0, 1.7, 1234]

    for xshift, yshift, zshift in  itertools.product(shifts, shifts, shifts):
        copygroup = copy.deepcopy(group)
        copygroup.shift(xshift, yshift, zshift)
        for loadid in LOAD:
            expected = copy.deepcopy(loads[loadid])
            expected.shift(xshift, yshift, zshift)
            assert getattr(copygroup, loadid) == expected
    

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        group = LoadGroup()
        loads = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref)
            group = group + load
            loads[loadid] = load
        
        group.to_reference(ref)
        shifts = [-23.5, -0.6, 0, 1.7, 1234]

        for xshift, yshift, zshift in  itertools.product(shifts, shifts, shifts):
            copygroup = copy.deepcopy(group)
            copygroup.shift(xshift, yshift, zshift)
            for loadid in LOAD:
                expected = copy.deepcopy(loads[loadid])
                expected.shift(xshift, yshift, zshift)

                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == expected.reference


def test_xshift():
    '''Test xshift method from LoadGroup class
    '''

    group = LoadGroup()
    loads = {}
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                       moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    
    shifts = [-23.5, -0.6, 0, 1.7, 1234]

    for shift in shifts:
        copygroup = copy.deepcopy(group)
        copygroup.xshift(shift)
        for loadid in LOAD:
            expected = copy.deepcopy(loads[loadid])
            expected.xshift(shift)
            assert getattr(copygroup, loadid) == expected
    

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        group = LoadGroup()
        loads = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref)
            group = group + load
            loads[loadid] = load
        
        group.to_reference(ref)
        shifts = [-23.5, -0.6, 0, 1.7, 1234]

        for shift in shifts:
            copygroup = copy.deepcopy(group)
            copygroup.xshift(shift)
            for loadid in LOAD:
                expected = copy.deepcopy(loads[loadid])
                expected.xshift(shift)

                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == expected.reference


def test_yshift():
    '''Test yshift method from LoadGroup class
    '''

    group = LoadGroup()
    loads = {}
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                       moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    
    shifts = [-23.5, -0.6, 0, 1.7, 1234]

    for shift in shifts:
        copygroup = copy.deepcopy(group)
        copygroup.yshift(shift)
        for loadid in LOAD:
            expected = copy.deepcopy(loads[loadid])
            expected.yshift(shift)
            assert getattr(copygroup, loadid) == expected
    

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        group = LoadGroup()
        loads = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref)
            group = group + load
            loads[loadid] = load
        
        group.to_reference(ref)
        shifts = [-23.5, -0.6, 0, 1.7, 1234]

        for shift in shifts:
            copygroup = copy.deepcopy(group)
            copygroup.yshift(shift)
            for loadid in LOAD:
                expected = copy.deepcopy(loads[loadid])
                expected.yshift(shift)

                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == expected.reference


def test_zshift():
    '''Test zshift method from LoadGroup class
    '''

    group = LoadGroup()
    loads = {}
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                       moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    
    shifts = [-23.5, -0.6, 0, 1.7, 1234]

    for shift in shifts:
        copygroup = copy.deepcopy(group)
        copygroup.zshift(shift)
        for loadid in LOAD:
            expected = copy.deepcopy(loads[loadid])
            expected.zshift(shift)
            assert getattr(copygroup, loadid) == expected
    

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        group = LoadGroup()
        loads = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref)
            group = group + load
            loads[loadid] = load
        
        group.to_reference(ref)
        shifts = [-23.5, -0.6, 0, 1.7, 1234]

        for shift in shifts:
            copygroup = copy.deepcopy(group)
            copygroup.zshift(shift)
            for loadid in LOAD:
                expected = copy.deepcopy(loads[loadid])
                expected.zshift(shift)

                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == expected.reference


def test_shift_ref():
    '''Test shift_ref method from LoadGroup class
    '''

    group = LoadGroup()
    loads = {}
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                       moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    
    shifts = [-23.5, -0.6, 0, 1.7, 1234]

    for xshift, yshift, zshift in  itertools.product(shifts, shifts, shifts):
        copygroup = copy.deepcopy(group)
        copygroup.shift_ref(xshift, yshift, zshift)
        for loadid in LOAD:
            expected = copy.deepcopy(loads[loadid])
            expected.shift_ref(xshift, yshift, zshift)
            assert getattr(copygroup, loadid) == expected
    

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        group = LoadGroup()
        loads = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref)
            group = group + load
            loads[loadid] = load
        
        group.to_reference(ref)
        shifts = [-23.5, -0.6, 0, 1.7, 1234]

        for xshift, yshift, zshift in  itertools.product(shifts, shifts, shifts):
            copygroup = copy.deepcopy(group)
            copygroup.shift_ref(xshift, yshift, zshift)
            for loadid in LOAD:
                expected = copy.deepcopy(loads[loadid])
                expected.shift_ref(xshift, yshift, zshift)

                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == expected.reference


def test_xshift_ref():
    '''Test xshift_ref method from LoadGroup class
    '''

    group = LoadGroup()
    loads = {}
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                       moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    
    shifts = [-23.5, -0.6, 0, 1.7, 1234]

    for shift in shifts:
        copygroup = copy.deepcopy(group)
        copygroup.xshift_ref(shift)
        for loadid in LOAD:
            expected = copy.deepcopy(loads[loadid])
            expected.xshift_ref(shift)
            assert getattr(copygroup, loadid) == expected
    

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        group = LoadGroup()
        loads = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref)
            group = group + load
            loads[loadid] = load
        
        group.to_reference(ref)
        shifts = [-23.5, -0.6, 0, 1.7, 1234]

        for shift in shifts:
            copygroup = copy.deepcopy(group)
            copygroup.xshift_ref(shift)
            for loadid in LOAD:
                expected = copy.deepcopy(loads[loadid])
                expected.xshift_ref(shift)

                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == expected.reference


def test_yshift_ref():
    '''Test yshift method from LoadGroup class
    '''

    group = LoadGroup()
    loads = {}
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                       moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    
    shifts = [-23.5, -0.6, 0, 1.7, 1234]

    for shift in shifts:
        copygroup = copy.deepcopy(group)
        copygroup.yshift_ref(shift)
        for loadid in LOAD:
            expected = copy.deepcopy(loads[loadid])
            expected.yshift_ref(shift)
            assert getattr(copygroup, loadid) == expected
    

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        group = LoadGroup()
        loads = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref)
            group = group + load
            loads[loadid] = load
        
        group.to_reference(ref)
        shifts = [-23.5, -0.6, 0, 1.7, 1234]

        for shift in shifts:
            copygroup = copy.deepcopy(group)
            copygroup.yshift_ref(shift)
            for loadid in LOAD:
                expected = copy.deepcopy(loads[loadid])
                expected.yshift_ref(shift)

                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == expected.reference


def test_zshift_ref():
    '''Test zshift_ref method from LoadGroup class
    '''

    group = LoadGroup()
    loads = {}
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                       moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    
    shifts = [-23.5, -0.6, 0, 1.7, 1234]

    for shift in shifts:
        copygroup = copy.deepcopy(group)
        copygroup.zshift_ref(shift)
        for loadid in LOAD:
            expected = copy.deepcopy(loads[loadid])
            expected.zshift_ref(shift)
            assert getattr(copygroup, loadid) == expected
    

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        group = LoadGroup()
        loads = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref)
            group = group + load
            loads[loadid] = load
        
        group.to_reference(ref)
        shifts = [-23.5, -0.6, 0, 1.7, 1234]

        for shift in shifts:
            copygroup = copy.deepcopy(group)
            copygroup.zshift_ref(shift)
            for loadid in LOAD:
                expected = copy.deepcopy(loads[loadid])
                expected.zshift_ref(shift)

                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == expected.reference


def test_rotate_along():
    '''Test rotate_along method from LoadGroup class
    '''

    group = LoadGroup()
    loads = {}
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                       moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    
    theta = np.linspace(-180, 180, 18+1)

    for lat, lon in itertools.product(theta, theta):                              
        direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                              np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                              np.sin(np.radians(lat))])
        for rot in theta:
            copygroup = copy.deepcopy(group)
            copygroup.rotate_along(direction, rot)
            for loadid in LOAD:
                expected = copy.deepcopy(loads[loadid])
                expected.rotate_along(direction, rot)
                assert getattr(copygroup, loadid) == expected
    

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        group = LoadGroup()
        loads = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref)
            group = group + load
            loads[loadid] = load
        
        group.to_reference(ref)

        for lat, lon in itertools.product(theta, theta):                              
            direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                                np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                                np.sin(np.radians(lat))])
            for rot in theta:
                copygroup = copy.deepcopy(group)
                copygroup.rotate_along(direction, rot)
                for loadid in LOAD:
                    expected = copy.deepcopy(loads[loadid])
                    expected.rotate_along(direction, rot)

                    assert getattr(copygroup, loadid).loadtype == expected.loadtype
                    assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                    assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                    assert getattr(copygroup, loadid).reference == expected.reference


def test_xrotate():
    '''Test xrotate method from LoadGroup class
    '''

    group = LoadGroup()
    loads = {}
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                       moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    
    theta = np.linspace(-180, 180, 18+1)

    for rot in theta:
        copygroup = copy.deepcopy(group)
        copygroup.xrotate(rot)
        for loadid in LOAD:
            expected = copy.deepcopy(loads[loadid])
            expected.xrotate(rot)
            assert getattr(copygroup, loadid) == expected
    

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        group = LoadGroup()
        loads = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref)
            group = group + load
            loads[loadid] = load
        
        group.to_reference(ref)

        for rot in theta:
            copygroup = copy.deepcopy(group)
            copygroup.xrotate(rot)
            for loadid in LOAD:
                expected = copy.deepcopy(loads[loadid])
                expected.xrotate(rot)

                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == expected.reference


def test_yrotate():
    '''Test yrotate method from LoadGroup class
    '''

    group = LoadGroup()
    loads = {}
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                       moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    
    theta = np.linspace(-180, 180, 18+1)

    for rot in theta:
        copygroup = copy.deepcopy(group)
        copygroup.yrotate(rot)
        for loadid in LOAD:
            expected = copy.deepcopy(loads[loadid])
            expected.yrotate(rot)
            assert getattr(copygroup, loadid) == expected
    

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        group = LoadGroup()
        loads = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref)
            group = group + load
            loads[loadid] = load
        
        group.to_reference(ref)

        for rot in theta:
            copygroup = copy.deepcopy(group)
            copygroup.yrotate(rot)
            for loadid in LOAD:
                expected = copy.deepcopy(loads[loadid])
                expected.yrotate(rot)

                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == expected.reference


def test_zrotate():
    '''Test zrotate method from LoadGroup class
    '''

    group = LoadGroup()
    loads = {}
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                       moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    
    theta = np.linspace(-180, 180, 18+1)

    for rot in theta:
        copygroup = copy.deepcopy(group)
        copygroup.zrotate(rot)
        for loadid in LOAD:
            expected = copy.deepcopy(loads[loadid])
            expected.zrotate(rot)
            assert getattr(copygroup, loadid) == expected
    

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        group = LoadGroup()
        loads = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref)
            group = group + load
            loads[loadid] = load
        
        group.to_reference(ref)

        for rot in theta:
            copygroup = copy.deepcopy(group)
            copygroup.zrotate(rot)
            for loadid in LOAD:
                expected = copy.deepcopy(loads[loadid])
                expected.zrotate(rot)

                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == expected.reference


def test_rotate_along_ref():
    '''Test rotate_along_ref method from LoadGroup class
    '''

    group = LoadGroup()
    loads = {}
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                       moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    
    theta = np.linspace(-180, 180, 18+1)

    for lat, lon in itertools.product(theta, theta):                              
        direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                              np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                              np.sin(np.radians(lat))])
        for rot in theta:
            copygroup = copy.deepcopy(group)
            copygroup.rotate_along_ref(direction, rot)
            for loadid in LOAD:
                expected = copy.deepcopy(loads[loadid])
                expected.rotate_along_ref(direction, rot)
                assert getattr(copygroup, loadid) == expected
    

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        group = LoadGroup()
        loads = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref)
            group = group + load
            loads[loadid] = load
        
        group.to_reference(ref)

        for lat, lon in itertools.product(theta, theta):                              
            direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                                np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                                np.sin(np.radians(lat))])
            for rot in theta:
                copygroup = copy.deepcopy(group)
                copygroup.rotate_along_ref(direction, rot)
                for loadid in LOAD:
                    expected = copy.deepcopy(loads[loadid])
                    expected.rotate_along_ref(direction, rot)

                    assert getattr(copygroup, loadid).loadtype == expected.loadtype
                    assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                    assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                    assert getattr(copygroup, loadid).reference == expected.reference


def test_xrotate_ref():
    '''Test xrotate_ref method from LoadGroup class
    '''

    group = LoadGroup()
    loads = {}
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                       moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    
    theta = np.linspace(-180, 180, 18+1)

    for rot in theta:
        copygroup = copy.deepcopy(group)
        copygroup.xrotate_ref(rot)
        for loadid in LOAD:
            expected = copy.deepcopy(loads[loadid])
            expected.xrotate_ref(rot)
            assert getattr(copygroup, loadid) == expected
    

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        group = LoadGroup()
        loads = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref)
            group = group + load
            loads[loadid] = load
        
        group.to_reference(ref)

        for rot in theta:
            copygroup = copy.deepcopy(group)
            copygroup.xrotate_ref(rot)
            for loadid in LOAD:
                expected = copy.deepcopy(loads[loadid])
                expected.xrotate_ref(rot)

                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == expected.reference


def test_yrotate_ref():
    '''Test yrotate_ref method from LoadGroup class
    '''

    group = LoadGroup()
    loads = {}
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                       moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    
    theta = np.linspace(-180, 180, 18+1)

    for rot in theta:
        copygroup = copy.deepcopy(group)
        copygroup.yrotate_ref(rot)
        for loadid in LOAD:
            expected = copy.deepcopy(loads[loadid])
            expected.yrotate_ref(rot)
            assert getattr(copygroup, loadid) == expected
    

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        group = LoadGroup()
        loads = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref)
            group = group + load
            loads[loadid] = load
        
        group.to_reference(ref)

        for rot in theta:
            copygroup = copy.deepcopy(group)
            copygroup.yrotate_ref(rot)
            for loadid in LOAD:
                expected = copy.deepcopy(loads[loadid])
                expected.yrotate_ref(rot)

                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == expected.reference


def test_zrotate_ref():
    '''Test zrotate_ref method from LoadGroup class
    '''

    group = LoadGroup()
    loads = {}
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                       moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    
    theta = np.linspace(-180, 180, 18+1)

    for rot in theta:
        copygroup = copy.deepcopy(group)
        copygroup.zrotate_ref(rot)
        for loadid in LOAD:
            expected = copy.deepcopy(loads[loadid])
            expected.zrotate_ref(rot)
            assert getattr(copygroup, loadid) == expected
    

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        group = LoadGroup()
        loads = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref)
            group = group + load
            loads[loadid] = load
        
        group.to_reference(ref)

        for rot in theta:
            copygroup = copy.deepcopy(group)
            copygroup.zrotate_ref(rot)
            for loadid in LOAD:
                expected = copy.deepcopy(loads[loadid])
                expected.zrotate_ref(rot)

                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == expected.reference


def test_to_reference():
    '''Test to_reference method from LoadGroup class
    '''

    coord = [-84.2, 0, 3]
    thetas = [-180, -134.5, 0, 45, 105.3]
    for x0, y0, z0 in itertools.product(coord, coord, coord):
        for lat, lon, theta in itertools.product(thetas, thetas, thetas):
            ref = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                                    np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                                    np.sin(np.radians(lat))])
            ref.rotate_along(direction, theta)

            loads = {}
            group = LoadGroup()
            for loadid in LOAD:
                load = Load.fromarray(loadid, force=np.random.random(3),
                                            moment=np.random.random(3))
                group = group + load
                loads[loadid] = load
            
            copygroup = copy.deepcopy(group)
            copygroup.to_reference(ref)
            for loadid in LOAD:
                expected = copy.deepcopy(loads[loadid])
                expected.to_reference(ref)

                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == expected.reference
    

    # wrong input type
    loads = {}
    group = LoadGroup()
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                    moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    with pytest.raises(TypeError):
        expected.to_reference([2])
    with pytest.raises(TypeError):
        expected.to_reference(loads[LOAD[0]])
    with pytest.raises(TypeError):
        expected.to_reference(2)


def test_to_origin():
    '''Test to_origin method from LoadGroup class
    '''

    coord = [-84.2, 0, 3]
    thetas = [-180, -134.5, 0, 45, 105.3]
    for x0, y0, z0 in itertools.product(coord, coord, coord):
        for lat, lon, theta in itertools.product(thetas, thetas, thetas):
            ref = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                                    np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                                    np.sin(np.radians(lat))])
            ref.rotate_along(direction, theta)

            loads = {}
            group = LoadGroup()
            for loadid in LOAD:
                load = Load.fromarray(loadid, force=np.random.random(3),
                                      moment=np.random.random(3),
                                      reference=ref)
                group = group + load
                loads[loadid] = load
            
            copygroup = copy.deepcopy(group)
            copygroup.to_reference(ref)
            copygroup.to_origin()
            for loadid in LOAD:
                expected = copy.deepcopy(loads[loadid])
                expected.to_origin()
                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == expected.reference


def test_restetorigin():
    '''Test restetorigin method from LoadGroup class
    '''

    coord = [-84.2, 0, 3]
    thetas = [-180, -134.5, 0, 45, 105.3]
    for x0, y0, z0 in itertools.product(coord, coord, coord):
        for lat, lon, theta in itertools.product(thetas, thetas, thetas):
            ref = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                                    np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                                    np.sin(np.radians(lat))])
            ref.rotate_along(direction, theta)

            loads = {}
            group = LoadGroup()
            for loadid in LOAD:
                load = Load.fromarray(loadid, force=np.random.random(3),
                                      moment=np.random.random(3),
                                      reference=ref)
                group = group + load
                loads[loadid] = load
            
            copygroup = copy.deepcopy(group)
            copygroup.to_reference(ref)
            copygroup.resetorigin()
            for loadid in LOAD:
                expected = copy.deepcopy(loads[loadid])

                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == ReferenceFrame()


def test_add_load():
    '''Test __add_load__ method from LoadGroup class
    '''

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    refs = [ref1, ref2, ref3]
    for ref_group, ref_loads in itertools.product(refs, refs):
        group = LoadGroup()
        group.to_reference(ref_group)
        loads1 = {}
        loads2 = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref_loads)
            group = group + load
            loads1[loadid] = load

            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref_loads)
            group = group + load
            loads2[loadid] = load
        
        for loadid in LOAD:
            expected = loads1[loadid] + loads2[loadid]
            expected.to_reference(ref_group)

            assert getattr(group, loadid).loadtype == expected.loadtype
            assert pytest.approx(getattr(group, loadid).force, 0.000001) == expected.force
            assert pytest.approx(getattr(group, loadid).moment, 0.000001) == expected.moment
            assert getattr(group, loadid).reference == ref_group

    # wrong input type
    loads = {}
    group = LoadGroup()
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                    moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    with pytest.raises(TypeError):
        _ = '3' + group
    with pytest.raises(TypeError):
        _ = [3] +  group
    with pytest.raises(TypeError):
        _ = np.array([3]) + group
    with pytest.raises(TypeError):
        _ = group + '3'
    with pytest.raises(TypeError):
        _ = group + [3]
    with pytest.raises(TypeError):
        _ = group + np.array([3])


def test_add_loadgroup():
    '''Test __add_loadgroup__ method from LoadGroup class
    '''

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    refs = [ref1, ref2, ref3]
    for ref_group1, ref_group2 in itertools.product(refs, refs):
        group1 = LoadGroup()
        group1.to_reference(ref_group1)

        group2 = LoadGroup()
        group2.to_reference(ref_group2)

        loads1 = {}
        loads2 = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3))
            group1 = group1 + load
            loads1[loadid] = load

            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3))
            group2 = group2 + load
            loads2[loadid] = load
        
        groupa = group1 + group2
        groupb = group2 + group1
        for loadid in LOAD:
            expected = loads1[loadid] + loads2[loadid]

            expected.to_reference(ref_group1)
            assert getattr(groupa, loadid).loadtype == expected.loadtype
            assert pytest.approx(getattr(groupa, loadid).force, 0.000001) == expected.force
            assert pytest.approx(getattr(groupa, loadid).moment, 0.000001) == expected.moment
            assert getattr(groupa, loadid).reference == ref_group1


            expected.to_reference(ref_group2)
            assert getattr(groupb, loadid).loadtype == expected.loadtype
            assert pytest.approx(getattr(groupb, loadid).force, 0.000001) == expected.force
            assert pytest.approx(getattr(groupb, loadid).moment, 0.000001) == expected.moment
            assert getattr(groupb, loadid).reference == ref_group2
        
        # changes in the origina groups dont change the sum
        group1.to_origin()
        group2 = group2 + group2

        for loadid in LOAD:
            expected = loads1[loadid] + loads2[loadid]

            expected.to_reference(ref_group1)
            assert getattr(groupa, loadid).loadtype == expected.loadtype
            assert pytest.approx(getattr(groupa, loadid).force, 0.000001) == expected.force
            assert pytest.approx(getattr(groupa, loadid).moment, 0.000001) == expected.moment
            assert getattr(groupa, loadid).reference == ref_group1


            expected.to_reference(ref_group2)
            assert getattr(groupb, loadid).loadtype == expected.loadtype
            assert pytest.approx(getattr(groupb, loadid).force, 0.000001) == expected.force
            assert pytest.approx(getattr(groupb, loadid).moment, 0.000001) == expected.moment
            assert getattr(groupb, loadid).reference == ref_group2


def test_sub_load():
    '''Test __sub_load__ method from LoadGroup class
    '''

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    refs = [ref1, ref2, ref3]
    for ref_group, ref_loads in itertools.product(refs, refs):
        group = LoadGroup()
        group.to_reference(ref_group)
        loads1 = {}
        loads2 = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref_loads)
            group = group - load
            loads1[loadid] = load

            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref_loads)
            group = group - load
            loads2[loadid] = load
        
        for loadid in LOAD:
            expected = -1 * loads1[loadid] - loads2[loadid]
            expected.to_reference(ref_group)

            assert getattr(group, loadid).loadtype == expected.loadtype
            assert pytest.approx(getattr(group, loadid).force, 0.000001) == expected.force
            assert pytest.approx(getattr(group, loadid).moment, 0.000001) == expected.moment
            assert getattr(group, loadid).reference == ref_group

    # wrong input type
    loads = {}
    group = LoadGroup()
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                    moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    with pytest.raises(TypeError):
        _ = '3' - group
    with pytest.raises(TypeError):
        _ = [3] -  group
    with pytest.raises(TypeError):
        _ = np.array([3]) - group
    with pytest.raises(TypeError):
        _ = group - '3'
    with pytest.raises(TypeError):
        _ = group - [3]
    with pytest.raises(TypeError):
        _ = group - np.array([3])


def test_sub_loadgroup():
    '''Test __sub_loadgroup__ method from LoadGroup class
    '''

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    refs = [ref1, ref2, ref3]
    for ref_group1, ref_group2 in itertools.product(refs, refs):
        group1 = LoadGroup()
        group1.to_reference(ref_group1)

        group2 = LoadGroup()
        group2.to_reference(ref_group2)

        loads1 = {}
        loads2 = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3))
            group1 = group1 + load
            loads1[loadid] = load

            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3))
            group2 = group2 + load
            loads2[loadid] = load
        
        groupa = group1 - group2
        groupb = group2 - group1
        for loadid in LOAD:
            expected = loads1[loadid] - loads2[loadid]
            expected.to_reference(ref_group1)
            assert getattr(groupa, loadid).loadtype == expected.loadtype
            assert pytest.approx(getattr(groupa, loadid).force, 0.000001) == expected.force
            assert pytest.approx(getattr(groupa, loadid).moment, 0.000001) == expected.moment
            assert getattr(groupa, loadid).reference == ref_group1

            expected = loads2[loadid] - loads1[loadid]
            expected.to_reference(ref_group2)
            assert getattr(groupb, loadid).loadtype == expected.loadtype
            assert pytest.approx(getattr(groupb, loadid).force, 0.000001) == expected.force
            assert pytest.approx(getattr(groupb, loadid).moment, 0.000001) == expected.moment
            assert getattr(groupb, loadid).reference == ref_group2
        
        # changes in the origina groups dont change the sum
        group1.to_origin()
        group2 = group2 - group2

        for loadid in LOAD:
            expected = loads1[loadid] - loads2[loadid]
            expected.to_reference(ref_group1)
            assert getattr(groupa, loadid).loadtype == expected.loadtype
            assert pytest.approx(getattr(groupa, loadid).force, 0.000001) == expected.force
            assert pytest.approx(getattr(groupa, loadid).moment, 0.000001) == expected.moment
            assert getattr(groupa, loadid).reference == ref_group1

            expected = loads2[loadid] - loads1[loadid]
            expected.to_reference(ref_group2)
            assert getattr(groupb, loadid).loadtype == expected.loadtype
            assert pytest.approx(getattr(groupb, loadid).force, 0.000001) == expected.force
            assert pytest.approx(getattr(groupb, loadid).moment, 0.000001) == expected.moment
            assert getattr(groupb, loadid).reference == ref_group2


def test_mul():
    '''Test __mul__ and __rmul__ method from LoadGroup class
    '''

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    refs = [ref1, ref2, ref3]
    for ref_group, ref_loads in itertools.product(refs, refs):
        group = LoadGroup()
        group.to_reference(ref_group)
        loads = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref_loads)
            group = group + load
            loads[loadid] = load
        
        for val in [0, 0.34, 3, -23, -6.5, 1235.23]:
            copygroup = copy.deepcopy(group)
            copygroup = val * copygroup
            for loadid in LOAD:
                expected = val * loads[loadid]
                expected.to_reference(ref_group)
                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == ref_group

                # original not affected
                expected = loads[loadid]
                expected.to_reference(ref_group)
                assert getattr(group, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(group, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(group, loadid).moment, 0.000001) == expected.moment
                assert getattr(group, loadid).reference == ref_group
            
            copygroup = copy.deepcopy(group)
            copygroup = copygroup * val
            for loadid in LOAD:
                expected = val * loads[loadid]
                expected.to_reference(ref_group)

                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == ref_group

                # original not affected
                expected = loads[loadid]
                expected.to_reference(ref_group)
                assert getattr(group, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(group, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(group, loadid).moment, 0.000001) == expected.moment
                assert getattr(group, loadid).reference == ref_group

    # wrong input type
    loads = {}
    group = LoadGroup()
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                    moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    with pytest.raises(TypeError):
        _ = '3' * group
    with pytest.raises(TypeError):
        _ = [3] * group
    with pytest.raises(TypeError):
        _ = np.array([3]) * group
    with pytest.raises(TypeError):
        _ = group * '3'
    with pytest.raises(TypeError):
        _ = group * [3]
    with pytest.raises(TypeError):
        _ = group * np.array([3])


def test_truediv():
    '''Test __truediv__ method from LoadGroup class
    '''

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    refs = [ref1, ref2, ref3]
    for ref_group, ref_loads in itertools.product(refs, refs):
        group = LoadGroup()
        group.to_reference(ref_group)
        loads = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref_loads)
            group = group + load
            loads[loadid] = load
        
        for val in [0.34, 3, -23, -6.5, 1235.23]:
            copygroup = copy.deepcopy(group)
            copygroup = copygroup / val
            for loadid in LOAD:
                expected = loads[loadid] / val
                expected.to_reference(ref_group)
                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == ref_group

                # original not affected
                expected = loads[loadid]
                expected.to_reference(ref_group)
                assert getattr(group, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(group, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(group, loadid).moment, 0.000001) == expected.moment
                assert getattr(group, loadid).reference == ref_group
    
    # wrong input type
    loads = {}
    group = LoadGroup()
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                    moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    with pytest.raises(TypeError):
        _ = group / '3'
    with pytest.raises(TypeError):
        _ = group / [3]
    with pytest.raises(TypeError):
        _ = group / np.array([3])
    with pytest.raises(ValueError):
        _ = group / 0


def test_iadd_load():
    '''Test __iadd_load__ method from LoadGroup class
    '''

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    refs = [ref1, ref2, ref3]
    for ref_group, ref_loads in itertools.product(refs, refs):
        group = LoadGroup()
        group.to_reference(ref_group)
        loads1 = {}
        loads2 = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref_loads)
            group += load
            loads1[loadid] = load

            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref_loads)
            group += load
            loads2[loadid] = load
        
        for loadid in LOAD:
            expected = loads1[loadid] + loads2[loadid]
            expected.to_reference(ref_group)

            assert getattr(group, loadid).loadtype == expected.loadtype
            assert pytest.approx(getattr(group, loadid).force, 0.000001) == expected.force
            assert pytest.approx(getattr(group, loadid).moment, 0.000001) == expected.moment
            assert getattr(group, loadid).reference == ref_group

    # wrong input type
    loads = {}
    group = LoadGroup()
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                    moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    with pytest.raises(TypeError):
        group += '3' 
    with pytest.raises(TypeError):
        group += [3]  
    with pytest.raises(TypeError):
        group += np.array([3])


def test_iadd_loadgroup():
    '''Test __iadd_loadgroup__ method from LoadGroup class
    '''

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    refs = [ref1, ref2, ref3]
    for ref_group1, ref_group2 in itertools.product(refs, refs):
        group1 = LoadGroup()
        group1.to_reference(ref_group1)

        group2 = LoadGroup()
        group2.to_reference(ref_group2)

        loads1 = {}
        loads2 = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3))
            group1 += load
            loads1[loadid] = load

            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3))
            group2 += load
            loads2[loadid] = load
        
        groupa = group1
        groupa += group2
        groupb = group2
        groupb += group1
        for loadid in LOAD:
            expected = loads1[loadid] + loads2[loadid]

            expected.to_reference(ref_group1)
            assert getattr(groupa, loadid).loadtype == expected.loadtype
            assert pytest.approx(getattr(groupa, loadid).force, 0.000001) == expected.force
            assert pytest.approx(getattr(groupa, loadid).moment, 0.000001) == expected.moment
            assert getattr(groupa, loadid).reference == ref_group1

            expected.to_reference(ref_group2)
            assert getattr(groupb, loadid).loadtype == expected.loadtype
            assert pytest.approx(getattr(groupb, loadid).force, 0.000001) == expected.force
            assert pytest.approx(getattr(groupb, loadid).moment, 0.000001) == expected.moment
            assert getattr(groupb, loadid).reference == ref_group2
        
        # changes in the origina groups dont change the sum
        group1.to_origin()
        group2 = group2 + group2

        for loadid in LOAD:
            expected = loads1[loadid] + loads2[loadid]

            expected.to_reference(ref_group1)
            assert getattr(groupa, loadid).loadtype == expected.loadtype
            assert pytest.approx(getattr(groupa, loadid).force, 0.000001) == expected.force
            assert pytest.approx(getattr(groupa, loadid).moment, 0.000001) == expected.moment
            assert getattr(groupa, loadid).reference == ref_group1


            expected.to_reference(ref_group2)
            assert getattr(groupb, loadid).loadtype == expected.loadtype
            assert pytest.approx(getattr(groupb, loadid).force, 0.000001) == expected.force
            assert pytest.approx(getattr(groupb, loadid).moment, 0.000001) == expected.moment
            assert getattr(groupb, loadid).reference == ref_group2


def test_isub_load():
    '''Test __isub_load__ method from LoadGroup class
    '''

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    refs = [ref1, ref2, ref3]
    for ref_group, ref_loads in itertools.product(refs, refs):
        group = LoadGroup()
        group.to_reference(ref_group)
        loads1 = {}
        loads2 = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref_loads)
            group -= load
            loads1[loadid] = load

            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref_loads)
            group -= load
            loads2[loadid] = load
        
        for loadid in LOAD:
            expected = -1 * loads1[loadid] - loads2[loadid]
            expected.to_reference(ref_group)

            assert getattr(group, loadid).loadtype == expected.loadtype
            assert pytest.approx(getattr(group, loadid).force, 0.000001) == expected.force
            assert pytest.approx(getattr(group, loadid).moment, 0.000001) == expected.moment
            assert getattr(group, loadid).reference == ref_group

    # wrong input type
    loads = {}
    group = LoadGroup()
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                    moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    with pytest.raises(TypeError):
        group -= '3' 
    with pytest.raises(TypeError):
        group -= [3]
    with pytest.raises(TypeError):
        group -= np.array([3])


def test_isub_loadgroup():
    '''Test __isub_loadgroup__ method from LoadGroup class
    '''

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    refs = [ref1, ref2, ref3]
    for ref_group1, ref_group2 in itertools.product(refs, refs):
        group1 = LoadGroup()
        group1.to_reference(ref_group1)

        group2 = LoadGroup()
        group2.to_reference(ref_group2)

        loads1 = {}
        loads2 = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3))
            group1 = group1 + load
            loads1[loadid] = load

            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3))
            group2 = group2 + load
            loads2[loadid] = load
        
        groupa = group1
        groupa -= group2
        groupb = group2
        groupb -= group1
        for loadid in LOAD:
            expected = loads1[loadid] - loads2[loadid]
            expected.to_reference(ref_group1)
            assert getattr(groupa, loadid).loadtype == expected.loadtype
            assert pytest.approx(getattr(groupa, loadid).force, 0.000001) == expected.force
            assert pytest.approx(getattr(groupa, loadid).moment, 0.000001) == expected.moment
            assert getattr(groupa, loadid).reference == ref_group1

            expected = loads2[loadid] - loads1[loadid]
            expected.to_reference(ref_group2)
            assert getattr(groupb, loadid).loadtype == expected.loadtype
            assert pytest.approx(getattr(groupb, loadid).force, 0.000001) == expected.force
            assert pytest.approx(getattr(groupb, loadid).moment, 0.000001) == expected.moment
            assert getattr(groupb, loadid).reference == ref_group2
        
        # changes in the origina groups dont change the sum
        group1.to_origin()
        group2 = group2 - group2

        for loadid in LOAD:
            expected = loads1[loadid] - loads2[loadid]
            expected.to_reference(ref_group1)
            assert getattr(groupa, loadid).loadtype == expected.loadtype
            assert pytest.approx(getattr(groupa, loadid).force, 0.000001) == expected.force
            assert pytest.approx(getattr(groupa, loadid).moment, 0.000001) == expected.moment
            assert getattr(groupa, loadid).reference == ref_group1

            expected = loads2[loadid] - loads1[loadid]
            expected.to_reference(ref_group2)
            assert getattr(groupb, loadid).loadtype == expected.loadtype
            assert pytest.approx(getattr(groupb, loadid).force, 0.000001) == expected.force
            assert pytest.approx(getattr(groupb, loadid).moment, 0.000001) == expected.moment
            assert getattr(groupb, loadid).reference == ref_group2


def test_imul():
    '''Test __imul__ and __irmul__ method from LoadGroup class
    '''

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    refs = [ref1, ref2, ref3]
    for ref_group, ref_loads in itertools.product(refs, refs):
        group = LoadGroup()
        group.to_reference(ref_group)
        loads = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref_loads)
            group = group + load
            loads[loadid] = load
        
        for val in [0, 0.34, 3, -23, -6.5, 1235.23]:
            copygroup = copy.deepcopy(group)
            copygroup *= val
            for loadid in LOAD:
                expected = val * loads[loadid]
                expected.to_reference(ref_group)
                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == ref_group

                # original not affected
                expected = loads[loadid]
                expected.to_reference(ref_group)
                assert getattr(group, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(group, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(group, loadid).moment, 0.000001) == expected.moment
                assert getattr(group, loadid).reference == ref_group
            
    # wrong input type
    loads = {}
    group = LoadGroup()
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                    moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    with pytest.raises(TypeError):
        group *= '3' 
    with pytest.raises(TypeError):
        group *= [3] 
    with pytest.raises(TypeError):
        group *= np.array([3])


def test_itruediv():
    '''Test __itruediv__ method from LoadGroup class
    '''

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    refs = [ref1, ref2, ref3]
    for ref_group, ref_loads in itertools.product(refs, refs):
        group = LoadGroup()
        group.to_reference(ref_group)
        loads = {}
        for loadid in LOAD:
            load = Load.fromarray(loadid, force=np.random.random(3),
                                  moment=np.random.random(3), reference=ref_loads)
            group = group + load
            loads[loadid] = load
        
        for val in [0.34, 3, -23, -6.5, 1235.23]:
            copygroup = copy.deepcopy(group)
            copygroup /= val
            for loadid in LOAD:
                expected = loads[loadid] / val
                expected.to_reference(ref_group)
                assert getattr(copygroup, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(copygroup, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(copygroup, loadid).moment, 0.000001) == expected.moment
                assert getattr(copygroup, loadid).reference == ref_group

                # original not affected
                expected = loads[loadid]
                expected.to_reference(ref_group)
                assert getattr(group, loadid).loadtype == expected.loadtype
                assert pytest.approx(getattr(group, loadid).force, 0.000001) == expected.force
                assert pytest.approx(getattr(group, loadid).moment, 0.000001) == expected.moment
                assert getattr(group, loadid).reference == ref_group
    
    # wrong input type
    loads = {}
    group = LoadGroup()
    for loadid in LOAD:
        load = Load.fromarray(loadid, force=np.random.random(3),
                                    moment=np.random.random(3))
        group = group + load
        loads[loadid] = load
    with pytest.raises(TypeError):
        group /= '3'
    with pytest.raises(TypeError):
        group /= [3]
    with pytest.raises(TypeError):
        group /= np.array([3])
    with pytest.raises(ValueError):
        group /= 0