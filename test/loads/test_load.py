'''Module for testing the Load and Load classes in cirsoc_402.loads
'''

import copy
import numpy as np
import pytest
import itertools

from cirsoc_402.load import ReferenceFrame
from cirsoc_402.load.loadclass import Load
from cirsoc_402.load.loadclass import GenericLoad
from cirsoc_402.load import Load
from cirsoc_402.constants import LOAD


def test_init():
    '''Test __init__ method from Load class
    '''

    load = Load('D')
    reference = ReferenceFrame()
    assert load.loadtype == 'D'
    assert load.name == ''
    assert all(load.force == np.array([0, 0, 0]))
    assert all(load.moment == np.array([0, 0, 0]))
    assert load.reference == reference


    vals = [-123, 0, 0.5, 1, 101235]
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for f1, f2, f3 in itertools.product(vals, vals, vals):
        for m1, m2, m3 in itertools.product(vals, vals, vals):
            for ref in [ref1, ref2, ref3]:
                load = Load('L', xforce=f1, yforce=f2, zforce=f3,
                                 xmoment=m1, ymoment=m2, zmoment=m3,
                                 reference=ref)
                assert load.loadtype == 'L'
                assert load.name == ''
                assert all(load.force == np.array([f1, f2, f3]))
                assert all(load.moment == np.array([m1, m2, m3]))
                assert load.reference == ref
    
    # nan behavior
    reference = ReferenceFrame()
    load = Load('S', xforce=np.nan)
    assert load.loadtype == 'S'
    assert load.name == ''
    assert np.isnan(load.force[0]) == True
    assert load.force[1] == 0
    assert load.force[2] == 0
    assert all(load.moment == np.array([0, 0, 0]))
    assert load.reference == reference


    load = Load('S', yforce=np.nan, zforce=np.nan)
    assert load.loadtype == 'S'
    assert load.name == ''
    assert np.isnan(load.force[1]) == True
    assert np.isnan(load.force[2]) == True
    assert load.force[0] == 0
    assert all(load.moment == np.array([0, 0, 0]))
    assert load.reference == reference

    load = Load('S', xforce=1, yforce=np.nan, xmoment=np.nan, ymoment=2, zmoment=3)
    assert load.loadtype == 'S'
    assert load.name == ''
    assert load.force[0] == 1
    assert np.isnan(load.force[1]) == True
    assert load.force[2] == 0
    assert np.isnan(load.moment[0]) == True
    assert load.moment[1] == 2
    assert load.moment[2] == 3
    assert load.reference == reference

    # wrong input type
    with pytest.raises(TypeError):
        load = Load('L', xforce=[2])
    with pytest.raises(TypeError):
        load = Load('L', xforce=[2], yforce='1')
    with pytest.raises(TypeError):
        load = Load('L', xmoment=[2])
    with pytest.raises(TypeError):
        load = Load('L', reference=[2])
    with pytest.raises(TypeError):
        load = Load(['asd'])
    with pytest.raises(TypeError):
        load = Load(2)
    with pytest.raises(ValueError):
        load = Load('XX')
    with pytest.raises(TypeError):
        load = Load('D', name=3)
    with pytest.raises(TypeError):
        load = Load('D', name=['xasd'])


def test_repr():
    '''Test __repr__ method from Load class
    '''

    vals = [-123, 0, 0.5, 1, 101235, np.nan]
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for loadtype in ['D', 'L', 'S']:
        for name in ['', 'a', 'assc']:
            for f1, f2, f3 in itertools.product(vals, vals, vals):
                for m1, m2, m3 in itertools.product(vals, vals, vals):
                    for ref in [ref1, ref2, ref3]:
                        load = Load(loadtype, name=name, xforce=f1, yforce=f2,
                                    zforce=f3, xmoment=m1, ymoment=m2, zmoment=m3,
                                    reference=ref)
                        
                        txt = name + " (" + loadtype + ")\n"
                        txt += "F = ({:.2f}, {:.2f}, {:.2f})\n".format(f1, f2, f3)
                        txt += "M = ({:.2f}, {:.2f}, {:.2f})\n".format(m1, m2, m3)
                        txt += "At reference frame:\n"
                        txt += "R = ({:.2f}, {:.2f}, {:.2f})\n".format(ref.origin[0], ref.origin[1], ref.origin[2])
                        txt += "ex = ({:.2f}, {:.2f}, {:.2f})\n".format(ref.xversor[0], ref.xversor[1], ref.xversor[2])
                        txt += "ey = ({:.2f}, {:.2f}, {:.2f})\n".format(ref.yversor[0], ref.yversor[1], ref.yversor[2])
                        txt += "ez = ({:.2f}, {:.2f}, {:.2f})".format(ref.zversor[0], ref.zversor[1], ref.zversor[2])

                        assert txt == load.__repr__()


def test_eq():
    '''Test __eq__ method from Load class
    '''

    load1 = Load('W')
    load2 = Load('W')
    assert load1 == load2

    vals = [-123, 0, 0.5, 1, 101235]
    for f1, f2, f3 in itertools.product(vals, vals, vals):
        for m1, m2, m3 in itertools.product(vals, vals, vals):
            load1 = Load('W', xforce=f1, yforce=f2, zforce=f3,
                                xmoment=m1, ymoment=m2, zmoment=m3)
            load2 = Load('W', xforce=f1, yforce=f2, zforce=f3,
                                xmoment=m1, ymoment=m2, zmoment=m3)
            assert load1 == load2
    
    vals1 = [-123, 0,  101235]
    vals2 = [-122, 0.3, 1235]
    for f11, f21, f31 in itertools.product(vals1, vals1, vals1):
        for m11, m21, m31 in itertools.product(vals1, vals1, vals1):
                for f12, f22, f32 in itertools.product(vals2, vals2, vals2):
                    for m12, m22, m32 in itertools.product(vals2, vals2, vals2):
                        load1 = Load('D', xforce=f11, yforce=f21, zforce=f31,
                                            xmoment=m11, ymoment=m21, zmoment=m31)
                        load2 = Load('D', xforce=f12, yforce=f22, zforce=f32,
                                            xmoment=m12, ymoment=m22, zmoment=m32)
                        assert not(load1 == load2)

    load1 = Load('D')
    load2 = Load('L')
    assert not(load1 == load2)

    load1 = Load('D', xforce=1, yforce=2, zforce=3, xmoment=4, ymoment=5,
                 zmoment=6)
    load2 = Load('L', xforce=1, yforce=2, zforce=3, xmoment=4, ymoment=5,
                 zmoment=6)
    assert not(load1 == load2)

    load1 = Load('D', name='a')
    load2 = Load('D', name='b')
    assert not(load1 == load2)

    load1 = Load('D', xforce=1, yforce=2, zforce=3, xmoment=4, ymoment=5,
                 zmoment=6, name='a')
    load2 = Load('D', xforce=1, yforce=2, zforce=3, xmoment=4, ymoment=5,
                 zmoment=6, name='b')
    assert not(load1 == load2)


    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    load1 = Load('D', reference=ref1)
    load2 = Load('D', reference=ref2)
    assert not(load1 == load2)


def test_fromarray():
    '''Test fromarray method from Load class
    '''

    vals = [-123, 0, 0.5, 1, 101235]
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for f1, f2, f3 in itertools.product(vals, vals, vals):
        for m1, m2, m3 in itertools.product(vals, vals, vals):
            load = Load.fromarray('D', name='test', force=[f1, f2, f3],
                                         moment=[m1, m2, m3])
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert all(load.force == np.array([f1, f2, f3]))
            assert all(load.moment == np.array([m1, m2, m3]))
            assert load.reference == ReferenceFrame()

            load = Load.fromarray('D', name='test', force=[f1, f2, f3])
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert all(load.force == np.array([f1, f2, f3]))
            assert all(load.moment == np.array([0, 0, 0]))
            assert load.reference == ReferenceFrame()

            load = Load.fromarray('D', name='test', moment=[m1, m2, m3])
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert all(load.force == np.array([0, 0, 0]))
            assert all(load.moment == np.array([m1, m2, m3]))
            assert load.reference == ReferenceFrame()

            for ref in [ref1, ref2, ref3]:
                load = Load.fromarray('D', name='test',  force=[f1, f2, f3],
                                             moment=[m1, m2, m3],
                                             reference=ref)
                assert load.loadtype == 'D'
                assert load.name == 'test'
                assert all(load.force == np.array([f1, f2, f3]))
                assert all(load.moment == np.array([m1, m2, m3]))
                assert load.reference == ref

                load = Load.fromarray('D', name='test', force=[f1, f2, f3],
                                            reference=ref)
                assert load.loadtype == 'D'
                assert load.name == 'test'
                assert all(load.force == np.array([f1, f2, f3]))
                assert all(load.moment == np.array([0, 0, 0]))
                assert load.reference == ref

                load = Load.fromarray('D', name='test', moment=[m1, m2, m3],
                                            reference=ref)
                assert load.loadtype == 'D'
                assert load.name == 'test'
                assert all(load.force == np.array([0, 0, 0]))
                assert all(load.moment == np.array([m1, m2, m3]))
                assert load.reference == ref
    
    # nan behavior
    reference = ReferenceFrame()
    load = Load.fromarray('D', name='test', force=[np.nan, np.nan, np.nan])
    assert load.loadtype == 'D'
    assert load.name == 'test'
    assert np.isnan(load.force[0]) == True
    assert np.isnan(load.force[1]) == True
    assert np.isnan(load.force[2]) == True
    assert all(load.moment == np.array([0, 0, 0]))
    assert load.reference == reference

    reference = ReferenceFrame()
    load = Load.fromarray('D', name='test', force=[np.nan, 2, 3.4])
    assert load.loadtype == 'D'
    assert load.name == 'test'
    assert np.isnan(load.force[0]) == True
    assert load.force[1] == 2
    assert load.force[2] == 3.4
    assert all(load.moment == np.array([0, 0, 0]))
    assert load.reference == reference

    reference = ReferenceFrame()
    load = Load.fromarray('D', name='test', force=[np.nan, 2, np.nan])
    assert load.loadtype == 'D'
    assert load.name == 'test'
    assert np.isnan(load.force[0]) == True
    assert load.force[1] == 2
    assert np.isnan(load.force[2]) == True
    assert all(load.moment == np.array([0, 0, 0]))
    assert load.reference == reference


    reference = ReferenceFrame()
    load = Load.fromarray('D', name='test', moment=[np.nan, np.nan, np.nan])
    assert load.loadtype == 'D'
    assert load.name == 'test'
    assert np.isnan(load.moment[0]) == True
    assert np.isnan(load.moment[1]) == True
    assert np.isnan(load.moment[2]) == True
    assert all(load.force == np.array([0, 0, 0]))
    assert load.reference == reference

    reference = ReferenceFrame()
    load = Load.fromarray('D', name='test', moment=[np.nan, 2, 3.4])
    assert load.loadtype == 'D'
    assert load.name == 'test'
    assert np.isnan(load.moment[0]) == True
    assert load.moment[1] == 2
    assert load.moment[2] == 3.4
    assert all(load.force == np.array([0, 0, 0]))
    assert load.reference == reference

    reference = ReferenceFrame()
    load = Load.fromarray('D', name='test', moment=[np.nan, 2, np.nan])
    assert load.loadtype == 'D'
    assert load.name == 'test'
    assert np.isnan(load.moment[0]) == True
    assert load.moment[1] == 2
    assert np.isnan(load.moment[2]) == True
    assert all(load.force == np.array([0, 0, 0]))
    assert load.reference == reference


    # wrong input type
    with pytest.raises(TypeError):
        load = Load.fromarray('D', name='test', force='asd')
    with pytest.raises(TypeError):
        load = Load.fromarray('D', name='test', moment='asd')
    with pytest.raises(TypeError):
        load = Load.fromarray('D', name='test', force=[2, 'a', 1])
    with pytest.raises(TypeError):
        load = Load.fromarray('D', name='test', moment=[2, 'a', 1])
    with pytest.raises(ValueError):
        load = Load.fromarray('D', name='test', moment=[2])
    with pytest.raises(ValueError):
        load = Load.fromarray('D', name='test', moment=[2, 3, 4, 5])
    with pytest.raises(ValueError):
        load = Load.fromarray('D', name='test', force=[2])
    with pytest.raises(ValueError):
        load = Load.fromarray('D', name='test', force=[2, 3, 4, 5])
    with pytest.raises(TypeError):
        load = Load.fromarray('D', name='test', reference='asd')
    with pytest.raises(TypeError):
        load = Load.fromarray(['asd'])
    with pytest.raises(TypeError):
        load = Load.fromarray(2)
    with pytest.raises(ValueError):
        load = Load.fromarray('XX')
    with pytest.raises(TypeError):
        load = Load.fromarray('D', name=3)
    with pytest.raises(TypeError):
        load = Load.fromarray('D', name=['xasd'])


def test_shift():
    '''Test shift method from Load class
    '''

    # no shift
    load1 = Load('D', name='test', xforce=2, zforce=4, ymoment=-1)
    load2 = copy.deepcopy(load1)
    load2.shift(0,0,0)
    assert load1 == load2

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        load1 = Load.fromarray('D', name='test', force=np.random.random(3),
                                      moment=np.random.random(3), reference=ref)
        load2 = copy.deepcopy(load1)
        load2.shift(0,0,0)
        assert load1 == load2


    # x axis shift
    force = np.ones(3)
    ref = ReferenceFrame()
    for shift in  [-23.5, -0.6, 0, 1.7, 1234]:
        moment = np.zeros(3)
        shiftedref = copy.deepcopy(ref)
        shiftedref.shift(shift, 0, 0)
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load.shift(shift, 0, 0)
        moment[1] = moment[1] + shift * force[2]
        moment[2] = moment[2] - shift * force[1]
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.000001) == force
        assert pytest.approx(load.moment, 0.000001) == moment
        assert load.reference == shiftedref

    # x axis shift from dispalced reference frame
    force = np.ones(3)
    ref = ReferenceFrame(xcoord=2, ycoord=-4.3, zcoord=12)
    for shift in  [-23.5, -0.6, 0, 1.7, 1234]:
        moment = np.zeros(3)
        shiftedref = copy.deepcopy(ref)
        shiftedref.shift(shift, 0, 0)
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load.shift(shift, 0, 0)
        moment[1] = moment[1] + shift * force[2]
        moment[2] = moment[2] - shift * force[1]
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.000001) == force
        assert pytest.approx(load.moment, 0.000001) == moment
        assert load.reference == shiftedref

    # x axis shift from rotated
    ref = ReferenceFrame()
    ref.rotate_along([1, 2, -4.3], 12.4)
    for shift in  [-23.5, -0.6, 0, 1.7, 1234]:
        shiftedref = copy.deepcopy(ref)
        shiftedref.shift(shift, 0, 0)
        load = Load.fromarray('D', name='test', force=np.ones(3), moment=np.zeros(3), reference=ref)
        
        # force and moments expressed in the origin system
        moment_o = np.zeros(3)
        force_o = load.force[0] * load.reference.xversor \
                + load.force[1] * load.reference.yversor \
                + load.force[2] * load.reference.zversor
        moment_o[1] = moment_o[1] + shift * force_o[2]
        moment_o[2] = moment_o[2] - shift * force_o[1]
        # force and moment expressed in the reference system
        force = shiftedref.o2r(force_o)
        moment = shiftedref.o2r(moment_o)

        load.shift(shift, 0, 0)
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.000001) == force
        assert pytest.approx(load.moment, 0.000001) == moment
        assert load.reference == shiftedref

    # x axis shift
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)

    shifts = [-23.5, -0.6, 0, 1.7, 1234]
    for ref in [ref1, ref2, ref3]:
        for shift in shifts:
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment,
                                        reference=ref)
        
            shiftedref = copy.deepcopy(ref)
            shiftedref.shift(shift, 0, 0)
            
            # force and moments expressed in the origin system
            moment_o = load.moment[0] * load.reference.xversor \
                    + load.moment[1] * load.reference.yversor \
                    + load.moment[2] * load.reference.zversor
            force_o = load.force[0] * load.reference.xversor \
                    + load.force[1] * load.reference.yversor \
                    + load.force[2] * load.reference.zversor
            moment_o[1] = moment_o[1] + shift * force_o[2]
            moment_o[2] = moment_o[2] - shift * force_o[1]
            # force and moment expressed in the reference system
            force = shiftedref.o2r(force_o)
            moment = shiftedref.o2r(moment_o)

            load.shift(shift, 0, 0)
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.000001) == force
            assert pytest.approx(load.moment, 0.000001) == moment
            assert load.reference == shiftedref

    # y axis shift
    force = np.ones(3)
    ref = ReferenceFrame()
    for shift in  [-23.5, -0.6, 0, 1.7, 1234]:
        moment = np.zeros(3)
        shiftedref = copy.deepcopy(ref)
        shiftedref.shift(0, shift, 0)
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load.shift(0, shift, 0)
        moment[0] = moment[0] - shift * force[2]
        moment[2] = moment[2] + shift * force[0]
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.000001) == force
        assert pytest.approx(load.moment, 0.000001) == moment
        assert load.reference == shiftedref

    # y axis shift from dispalced reference frame
    force = np.ones(3)
    ref = ReferenceFrame(xcoord=2, ycoord=-4.3, zcoord=12)
    for shift in  [-23.5, -0.6, 0, 1.7, 1234]:
        moment = np.zeros(3)
        shiftedref = copy.deepcopy(ref)
        shiftedref.shift(0, shift, 0)
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load.shift(0, shift, 0)
        moment[0] = moment[0] - shift * force[2]
        moment[2] = moment[2] + shift * force[0]
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.000001) == force
        assert pytest.approx(load.moment, 0.000001) == moment
        assert load.reference == shiftedref

    # y axis shift
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)

    shifts = [-23.5, -0.6, 0, 1.7, 1234]
    for ref in [ref1, ref2, ref3]:
        for shift in shifts:
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment,
                                        reference=ref)
        
            shiftedref = copy.deepcopy(ref)
            shiftedref.shift(0, shift, 0)
            
            # force and moments expressed in the origin system
            moment_o = load.moment[0] * load.reference.xversor \
                    + load.moment[1] * load.reference.yversor \
                    + load.moment[2] * load.reference.zversor
            force_o = load.force[0] * load.reference.xversor \
                    + load.force[1] * load.reference.yversor \
                    + load.force[2] * load.reference.zversor
            moment_o[0] = moment_o[0] - shift * force_o[2]
            moment_o[2] = moment_o[2] + shift * force_o[0]
            # force and moment expressed in the reference system
            force = shiftedref.o2r(force_o)
            moment = shiftedref.o2r(moment_o)

            load.shift(0, shift, 0)
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.000001) == force
            assert pytest.approx(load.moment, 0.000001) == moment
            assert load.reference == shiftedref
    
    # z axis shift
    force = np.ones(3)
    ref = ReferenceFrame()
    for shift in  [-23.5, -0.6, 0, 1.7, 1234]:
        moment = np.zeros(3)
        shiftedref = copy.deepcopy(ref)
        shiftedref.shift(0, 0, shift)
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load.shift(0, 0, shift)
        moment[0] = moment[0] + shift * force[1]
        moment[1] = moment[1] - shift * force[0]
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.000001) == force
        assert pytest.approx(load.moment, 0.000001) == moment
        assert load.reference == shiftedref

    # z axis shift from dispalced reference frame
    force = np.ones(3)
    ref = ReferenceFrame(xcoord=2, ycoord=-4.3, zcoord=12)
    for shift in  [-23.5, -0.6, 0, 1.7, 1234]:
        moment = np.zeros(3)
        shiftedref = copy.deepcopy(ref)
        shiftedref.shift(0, 0, shift)
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load.shift(0, 0, shift)
        moment[0] = moment[0] + shift * force[1]
        moment[1] = moment[1] - shift * force[0]
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.000001) == force
        assert pytest.approx(load.moment, 0.000001) == moment
        assert load.reference == shiftedref
    
    # z axis shift
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)

    shifts = [-23.5, -0.6, 0, 1.7, 1234]
    for ref in [ref1, ref2, ref3]:
        for shift in shifts:
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment,
                                        reference=ref)
        
            shiftedref = copy.deepcopy(ref)
            shiftedref.shift(0, 0, shift)
            
            # force and moments expressed in the origin system
            moment_o = load.moment[0] * load.reference.xversor \
                    + load.moment[1] * load.reference.yversor \
                    + load.moment[2] * load.reference.zversor
            force_o = load.force[0] * load.reference.xversor \
                    + load.force[1] * load.reference.yversor \
                    + load.force[2] * load.reference.zversor
            moment_o[0] = moment_o[0] + shift * force_o[1]
            moment_o[1] = moment_o[1] - shift * force_o[0]
            # force and moment expressed in the reference system
            force = shiftedref.o2r(force_o)
            moment = shiftedref.o2r(moment_o)

            load.shift(0, 0, shift)
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.000001) == force
            assert pytest.approx(load.moment, 0.000001) == moment
            assert load.reference == shiftedref
    

    # general shift
    force = np.ones(3)
    ref = ReferenceFrame()
    shifts = [-23.5, -0.6, 0, 1.7, 1234]
    for xshift, yshift, zshift in  itertools.product(shifts, shifts, shifts):
        moment = np.zeros(3)
        shiftedref = copy.deepcopy(ref)
        shiftedref.shift(xshift, yshift, zshift)
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load.shift(xshift, yshift, zshift)
        moment[1] = moment[1] + xshift * force[2]
        moment[2] = moment[2] - xshift * force[1]

        moment[0] = moment[0] - yshift * force[2]
        moment[2] = moment[2] + yshift * force[0]

        moment[0] = moment[0] + zshift * force[1]
        moment[1] = moment[1] - zshift * force[0]

        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.000001) == force
        assert pytest.approx(load.moment, 0.000001) == moment
        assert load.reference == shiftedref
    
    # general shift from dispalced reference frame
    force = np.ones(3)
    ref = ReferenceFrame(xcoord=2, ycoord=-4.3, zcoord=12)
    shifts = [-23.5, -0.6, 0, 1.7, 1234]
    for xshift, yshift, zshift in  itertools.product(shifts, shifts, shifts):
        moment = np.zeros(3)
        shiftedref = copy.deepcopy(ref)
        shiftedref.shift(xshift, yshift, zshift)
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load.shift(xshift, yshift, zshift)

        moment[1] = moment[1] + xshift * force[2]
        moment[2] = moment[2] - xshift * force[1]

        moment[0] = moment[0] - yshift * force[2]
        moment[2] = moment[2] + yshift * force[0]

        moment[0] = moment[0] + zshift * force[1]
        moment[1] = moment[1] - zshift * force[0]

        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.000001) == force
        assert pytest.approx(load.moment, 0.000001) == moment
        assert load.reference == shiftedref

    # general shift
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)

    shifts = [-23.5, -0.6, 0, 1.7, 1234]
    for ref in [ref1, ref2, ref3]:
        for xshift, yshift, zshift in  itertools.product(shifts, shifts, shifts):
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment,
                                        reference=ref)
        
            shiftedref = copy.deepcopy(ref)
            shiftedref.shift(xshift, yshift, zshift)
            
            # force and moments expressed in the origin system
            moment_o = load.moment[0] * load.reference.xversor \
                    + load.moment[1] * load.reference.yversor \
                    + load.moment[2] * load.reference.zversor
            force_o = load.force[0] * load.reference.xversor \
                    + load.force[1] * load.reference.yversor \
                    + load.force[2] * load.reference.zversor
            
            moment_o[1] = moment_o[1] + xshift * force_o[2]
            moment_o[2] = moment_o[2] - xshift * force_o[1]

            moment_o[0] = moment_o[0] - yshift * force_o[2]
            moment_o[2] = moment_o[2] + yshift * force_o[0]

            moment_o[0] = moment_o[0] + zshift * force_o[1]
            moment_o[1] = moment_o[1] - zshift * force_o[0]
            # force and moment expressed in the reference system
            force = shiftedref.o2r(force_o)
            moment = shiftedref.o2r(moment_o)

            load.shift(xshift, yshift, zshift)
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.000001) == force
            assert pytest.approx(load.moment, 0.000001) == moment
            assert load.reference == shiftedref


def test_xshift():
    '''Test xshift method from Load class
    '''

    # no shift
    load1 = Load('D', name='test', xforce=2, zforce=4, ymoment=-1)
    load2 = copy.deepcopy(load1)
    load2.xshift(0)
    assert load1 == load2

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        load1 = Load.fromarray('D', name='test', force=np.random.random(3),
                                      moment=np.random.random(3), reference=ref)
        load2 = copy.deepcopy(load1)
        load2.xshift(0)
        assert load1 == load2

    # x axis shift
    force = np.ones(3)
    ref = ReferenceFrame()
    for shift in  [-23.5, -0.6, 0, 1.7, 1234]:
        moment = np.zeros(3)
        shiftedref = copy.deepcopy(ref)
        shiftedref.xshift(shift)
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load.xshift(shift)
        moment[1] = moment[1] + shift * force[2]
        moment[2] = moment[2] - shift * force[1]
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.000001) == force
        assert pytest.approx(load.moment, 0.000001) == moment
        assert load.reference == shiftedref

    # x axis shift from dispalced reference frame
    force = np.ones(3)
    ref = ReferenceFrame(xcoord=2, ycoord=-4.3, zcoord=12)
    for shift in  [-23.5, -0.6, 0, 1.7, 1234]:
        moment = np.zeros(3)
        shiftedref = copy.deepcopy(ref)
        shiftedref.xshift(shift)
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load.xshift(shift)
        moment[1] = moment[1] + shift * force[2]
        moment[2] = moment[2] - shift * force[1]
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.000001) == force
        assert pytest.approx(load.moment, 0.000001) == moment
        assert load.reference == shiftedref

    # x axis shift from rotated
    ref = ReferenceFrame()
    ref.rotate_along([1, 2, -4.3], 12.4)
    for shift in  [-23.5, -0.6, 0, 1.7, 1234]:
        shiftedref = copy.deepcopy(ref)
        shiftedref.xshift(shift)
        load = Load.fromarray('D', name='test', force=np.ones(3), moment=np.zeros(3), reference=ref)
        
        # force and moments expressed in the origin system
        moment_o = np.zeros(3)
        force_o = load.force[0] * load.reference.xversor \
                + load.force[1] * load.reference.yversor \
                + load.force[2] * load.reference.zversor
        moment_o[1] = moment_o[1] + shift * force_o[2]
        moment_o[2] = moment_o[2] - shift * force_o[1]
        # force and moment expressed in the reference system
        force = shiftedref.o2r(force_o)
        moment = shiftedref.o2r(moment_o)

        load.xshift(shift)
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.000001) == force
        assert pytest.approx(load.moment, 0.000001) == moment
        assert load.reference == shiftedref

    # x axis shift
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)

    shifts = [-23.5, -0.6, 0, 1.7, 1234]
    for ref in [ref1, ref2, ref3]:
        for shift in shifts:
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment,
                                        reference=ref)
        
            shiftedref = copy.deepcopy(ref)
            shiftedref.xshift(shift)
            
            # force and moments expressed in the origin system
            moment_o = load.moment[0] * load.reference.xversor \
                    + load.moment[1] * load.reference.yversor \
                    + load.moment[2] * load.reference.zversor
            force_o = load.force[0] * load.reference.xversor \
                    + load.force[1] * load.reference.yversor \
                    + load.force[2] * load.reference.zversor
            moment_o[1] = moment_o[1] + shift * force_o[2]
            moment_o[2] = moment_o[2] - shift * force_o[1]
            # force and moment expressed in the reference system
            force = shiftedref.o2r(force_o)
            moment = shiftedref.o2r(moment_o)

            load.xshift(shift)
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.000001) == force
            assert pytest.approx(load.moment, 0.000001) == moment
            assert load.reference == shiftedref


def test_yshift():
    '''Test yshift method from Load class
    '''

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        load1 = Load.fromarray('D', name='test', force=np.random.random(3),
                                      moment=np.random.random(3), reference=ref)
        load2 = copy.deepcopy(load1)
        load2.yshift(0)
        assert load1 == load2

    # y axis shift
    force = np.ones(3)
    ref = ReferenceFrame()
    for shift in  [-23.5, -0.6, 0, 1.7, 1234]:
        moment = np.zeros(3)
        shiftedref = copy.deepcopy(ref)
        shiftedref.yshift(shift)
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load.yshift(shift)
        moment[0] = moment[0] - shift * force[2]
        moment[2] = moment[2] + shift * force[0]
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.000001) == force
        assert pytest.approx(load.moment, 0.000001) == moment
        assert load.reference == shiftedref

    # y axis shift from dispalced reference frame
    force = np.ones(3)
    ref = ReferenceFrame(xcoord=2, ycoord=-4.3, zcoord=12)
    for shift in  [-23.5, -0.6, 0, 1.7, 1234]:
        moment = np.zeros(3)
        shiftedref = copy.deepcopy(ref)
        shiftedref.yshift(shift)
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load.yshift(shift)
        moment[0] = moment[0] - shift * force[2]
        moment[2] = moment[2] + shift * force[0]
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.000001) == force
        assert pytest.approx(load.moment, 0.000001) == moment
        assert load.reference == shiftedref

    # y axis shift
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)

    shifts = [-23.5, -0.6, 0, 1.7, 1234]
    for ref in [ref1, ref2, ref3]:
        for shift in shifts:
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment,
                                        reference=ref)
        
            shiftedref = copy.deepcopy(ref)
            shiftedref.yshift(shift)
            
            # force and moments expressed in the origin system
            moment_o = load.moment[0] * load.reference.xversor \
                    + load.moment[1] * load.reference.yversor \
                    + load.moment[2] * load.reference.zversor
            force_o = load.force[0] * load.reference.xversor \
                    + load.force[1] * load.reference.yversor \
                    + load.force[2] * load.reference.zversor
            moment_o[0] = moment_o[0] - shift * force_o[2]
            moment_o[2] = moment_o[2] + shift * force_o[0]
            # force and moment expressed in the reference system
            force = shiftedref.o2r(force_o)
            moment = shiftedref.o2r(moment_o)

            load.yshift(shift)
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.000001) == force
            assert pytest.approx(load.moment, 0.000001) == moment
            assert load.reference == shiftedref

   
def test_zshift():
    '''Test zshift method from Load class
    '''

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        load1 = Load.fromarray('D', name='test', force=np.random.random(3),
                                      moment=np.random.random(3), reference=ref)
        load2 = copy.deepcopy(load1)
        load2.zshift(0)
        assert load1 == load2

    # z axis shift
    force = np.ones(3)
    ref = ReferenceFrame()
    for shift in  [-23.5, -0.6, 0, 1.7, 1234]:
        moment = np.zeros(3)
        shiftedref = copy.deepcopy(ref)
        shiftedref.zshift(shift)
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load.zshift(shift)
        moment[0] = moment[0] + shift * force[1]
        moment[1] = moment[1] - shift * force[0]
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.000001) == force
        assert pytest.approx(load.moment, 0.000001) == moment
        assert load.reference == shiftedref

    # z axis shift from dispalced reference frame
    force = np.ones(3)
    ref = ReferenceFrame(xcoord=2, ycoord=-4.3, zcoord=12)
    for shift in  [-23.5, -0.6, 0, 1.7, 1234]:
        moment = np.zeros(3)
        shiftedref = copy.deepcopy(ref)
        shiftedref.zshift(shift)
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load.zshift(shift)
        moment[0] = moment[0] + shift * force[1]
        moment[1] = moment[1] - shift * force[0]
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.000001) == force
        assert pytest.approx(load.moment, 0.000001) == moment
        assert load.reference == shiftedref
    
    # z axis shift
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)

    shifts = [-23.5, -0.6, 0, 1.7, 1234]
    for ref in [ref1, ref2, ref3]:
        for shift in shifts:
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment,
                                        reference=ref)
        
            shiftedref = copy.deepcopy(ref)
            shiftedref.zshift(shift)
            
            # force and moments expressed in the origin system
            moment_o = load.moment[0] * load.reference.xversor \
                    + load.moment[1] * load.reference.yversor \
                    + load.moment[2] * load.reference.zversor
            force_o = load.force[0] * load.reference.xversor \
                    + load.force[1] * load.reference.yversor \
                    + load.force[2] * load.reference.zversor
            moment_o[0] = moment_o[0] + shift * force_o[1]
            moment_o[1] = moment_o[1] - shift * force_o[0]
            # force and moment expressed in the reference system
            force = shiftedref.o2r(force_o)
            moment = shiftedref.o2r(moment_o)

            load.zshift(shift)
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.000001) == force
            assert pytest.approx(load.moment, 0.000001) == moment
            assert load.reference == shiftedref


def test_shift_ref():
    '''Test shift_ref method from Load class
    '''

    # no shift
    load1 = Load('D', name='test', xforce=2, zforce=4, ymoment=-1)
    load2 = copy.deepcopy(load1)
    load2.shift_ref(0,0,0)
    assert load1 == load2

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        load1 = Load.fromarray('D', name='test', force=np.random.random(3),
                                      moment=np.random.random(3), reference=ref)
        load2 = copy.deepcopy(load1)
        load2.shift_ref(0,0,0)
        assert load1 == load2

    # x axis shift
    force = np.ones(3)
    for ref in [ref1, ref2, ref3]:
        for shift in  [-23.5, -0.6, 0, 1.7, 1234]:
            moment = np.zeros(3)
            shiftedref = copy.deepcopy(ref)
            shiftedref.shift_ref(shift, 0, 0)
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load.shift_ref(shift, 0, 0)
            moment[1] = moment[1] + shift * force[2]
            moment[2] = moment[2] - shift * force[1]
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.000001) == force
            assert pytest.approx(load.moment, 0.000001) == moment
            assert load.reference == shiftedref
    
    coord = [-84.2, -0.4, 0, 3, 18.2]
    for xcoord, ycoord, zcoord in itertools.product(coord, coord, coord):
        ref = ReferenceFrame(xcoord=xcoord, ycoord=ycoord, zcoord=zcoord)
        ref.rotate_along(np.random.random(3), 360 * np.random.random() - 180)
        for shift in  [-23.5, -0.6, 0, 1.7, 1234]:
            force = 200 * np.random.random(3) - 100
            moment = 200 * np.random.random(3) - 100

            shiftedref = copy.deepcopy(ref)
            shiftedref.shift_ref(shift, 0, 0)
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load.shift_ref(shift, 0, 0)
            moment[1] = moment[1] + shift * force[2]
            moment[2] = moment[2] - shift * force[1]
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.000001) == force
            assert pytest.approx(load.moment, 0.000001) == moment
            assert load.reference == shiftedref
    
    
    # y axis shift
    coord = [-84.2, -0.4, 0, 3, 18.2]
    for xcoord, ycoord, zcoord in itertools.product(coord, coord, coord):
        ref = ReferenceFrame(xcoord=xcoord, ycoord=ycoord, zcoord=zcoord)
        ref.rotate_along(np.random.random(3), 360 * np.random.random() - 180)
        for shift in  [-23.5, -0.6, 0, 1.7, 1234]:
            force = 200 * np.random.random(3) - 100
            moment = 200 * np.random.random(3) - 100

            shiftedref = copy.deepcopy(ref)
            shiftedref.shift_ref(0, shift, 0)
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load.shift_ref(0, shift, 0)
            
            moment[0] = moment[0] - shift * force[2]
            moment[2] = moment[2] + shift * force[0]

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.000001) == force
            assert pytest.approx(load.moment, 0.000001) == moment
            assert load.reference == shiftedref
    
    # z axis shift
    coord = [-84.2, -0.4, 0, 3, 18.2]
    for xcoord, ycoord, zcoord in itertools.product(coord, coord, coord):
        ref = ReferenceFrame(xcoord=xcoord, ycoord=ycoord, zcoord=zcoord)
        ref.rotate_along(np.random.random(3), 360 * np.random.random() - 180)
        for shift in  [-23.5, -0.6, 0, 1.7, 1234]:
            force = 200 * np.random.random(3) - 100
            moment = 200 * np.random.random(3) - 100

            shiftedref = copy.deepcopy(ref)
            shiftedref.shift_ref(0, 0, shift)
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load.shift_ref(0, 0, shift)
            
            moment[0] = moment[0] + shift * force[1]
            moment[1] = moment[1] - shift * force[0]

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.000001) == force
            assert pytest.approx(load.moment, 0.000001) == moment
            assert load.reference == shiftedref
    
    # general shift
    coord = [-84.2, -0.4, 0, 3, 18.2]
    shifts = [-23.5, -0.6, 0, 1.7, 1234]
    for xcoord, ycoord, zcoord in itertools.product(coord, coord, coord):
        ref = ReferenceFrame(xcoord=xcoord, ycoord=ycoord, zcoord=zcoord)
        ref.rotate_along(np.random.random(3), 360 * np.random.random() - 180)
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            force = 200 * np.random.random(3) - 100
            moment = 200 * np.random.random(3) - 100

            shiftedref = copy.deepcopy(ref)
            shiftedref.shift_ref(xshift, yshift, zshift)
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load.shift_ref(xshift, yshift, zshift)

            moment[1] = moment[1] + xshift * force[2]
            moment[2] = moment[2] - xshift * force[1]
    
            moment[0] = moment[0] - yshift * force[2]
            moment[2] = moment[2] + yshift * force[0]
            
            moment[0] = moment[0] + zshift * force[1]
            moment[1] = moment[1] - zshift * force[0]

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.000001) == force
            assert pytest.approx(load.moment, 0.000001) == moment
            assert load.reference == shiftedref


def test_xshift_ref():
    '''Test xshift_ref method from Load class
    '''

    # no shift
    load1 = Load('D', name='test', xforce=2, zforce=4, ymoment=-1)
    load2 = copy.deepcopy(load1)
    load2.xshift_ref(0)
    assert load1 == load2

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        load1 = Load.fromarray('D', name='test', force=np.random.random(3),
                                      moment=np.random.random(3), reference=ref)
        load2 = copy.deepcopy(load1)
        load2.xshift_ref(0)
        assert load1 == load2

    coord = [-84.2, -0.4, 0, 3, 18.2]
    for xcoord, ycoord, zcoord in itertools.product(coord, coord, coord):
        ref = ReferenceFrame(xcoord=xcoord, ycoord=ycoord, zcoord=zcoord)
        ref.rotate_along(np.random.random(3), 360 * np.random.random() - 180)
        for shift in  [-23.5, -0.6, 0, 1.7, 1234]:
            force = 200 * np.random.random(3) - 100
            moment = 200 * np.random.random(3) - 100

            shiftedref = copy.deepcopy(ref)
            shiftedref.xshift_ref(shift)
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load.xshift_ref(shift)
            moment[1] = moment[1] + shift * force[2]
            moment[2] = moment[2] - shift * force[1]
            
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.000001) == force
            assert pytest.approx(load.moment, 0.000001) == moment
            assert load.reference == shiftedref


def test_yshift_ref():
    '''Test yshift_ref method from Load class
    '''

    # no shift
    load1 = Load('D', name='test', xforce=2, zforce=4, ymoment=-1)
    load2 = copy.deepcopy(load1)
    load2.yshift_ref(0)
    assert load1 == load2

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        load1 = Load.fromarray('D', name='test', force=np.random.random(3),
                                      moment=np.random.random(3), reference=ref)
        load2 = copy.deepcopy(load1)
        load2.yshift_ref(0)
        assert load1 == load2

    coord = [-84.2, -0.4, 0, 3, 18.2]
    for xcoord, ycoord, zcoord in itertools.product(coord, coord, coord):
        ref = ReferenceFrame(xcoord=xcoord, ycoord=ycoord, zcoord=zcoord)
        ref.rotate_along(np.random.random(3), 360 * np.random.random() - 180)
        for shift in  [-23.5, -0.6, 0, 1.7, 1234]:
            force = 200 * np.random.random(3) - 100
            moment = 200 * np.random.random(3) - 100

            shiftedref = copy.deepcopy(ref)
            shiftedref.yshift_ref(shift)
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load.yshift_ref(shift)
            
            moment[0] = moment[0] - shift * force[2]
            moment[2] = moment[2] + shift * force[0]

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.000001) == force
            assert pytest.approx(load.moment, 0.000001) == moment
            assert load.reference == shiftedref


def test_zshift_ref():
    '''Test xshift_ref method from Load class
    '''

    # no shift
    load1 = Load('D', name='test', xforce=2, zforce=4, ymoment=-1)
    load2 = copy.deepcopy(load1)
    load2.zshift_ref(0)
    assert load1 == load2

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        load1 = Load.fromarray('D', name='test', force=np.random.random(3),
                                      moment=np.random.random(3), reference=ref)
        load2 = copy.deepcopy(load1)
        load2.zshift_ref(0)
        assert load1 == load2
    
    coord = [-84.2, -0.4, 0, 3, 18.2]
    for xcoord, ycoord, zcoord in itertools.product(coord, coord, coord):
        ref = ReferenceFrame(xcoord=xcoord, ycoord=ycoord, zcoord=zcoord)
        ref.rotate_along(np.random.random(3), 360 * np.random.random() - 180)
        for shift in  [-23.5, -0.6, 0, 1.7, 1234]:
            force = 200 * np.random.random(3) - 100
            moment = 200 * np.random.random(3) - 100

            shiftedref = copy.deepcopy(ref)
            shiftedref.zshift_ref(shift)
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load.zshift_ref(shift)
            
            moment[0] = moment[0] + shift * force[1]
            moment[1] = moment[1] - shift * force[0]

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.000001) == force
            assert pytest.approx(load.moment, 0.000001) == moment
            assert load.reference == shiftedref


def test_rotate_along():
    '''Test rotate_along method from Load class
    '''
    # no shift
    load1 = Load('D', name='test', xforce=2, zforce=4, ymoment=-1)
    load2 = copy.deepcopy(load1)
    load2.rotate_along([1, 1, 1], 0)
    assert load1 == load2
    load2 = copy.deepcopy(load1)
    load2.rotate_along([1, 0, 0], 0)
    assert load1 == load2
    load2 = copy.deepcopy(load1)
    load2.rotate_along([0, 1, 0], 0)
    assert load1 == load2
    load2 = copy.deepcopy(load1)
    load2.rotate_along([0, 0, 1], 0)
    assert load1 == load2
    load2 = copy.deepcopy(load1)
    load2.rotate_along([1, 2, 3], 0)
    assert load1 == load2
    load2 = copy.deepcopy(load1)
    load2.rotate_along([-1, -1, -1], 0)
    assert load1 == load2
    load2 = copy.deepcopy(load1)
    load2.rotate_along([1, -1, -1], 0)
    assert load1 == load2


    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        load1 = Load.fromarray('D', name='test', force=np.random.random(3),
                                      moment=np.random.random(3), reference=ref)
        load2 = copy.deepcopy(load1)
        load2.rotate_along([1, 1, 1], 0)
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == load2.force
        assert pytest.approx(load1.moment, 0.000001) == load2.moment
        assert pytest.approx(load1.reference.origin, 0.000001) == load2.reference.origin
        assert pytest.approx(load1.reference.xversor, 0.000001) == load2.reference.xversor
        assert pytest.approx(load1.reference.yversor, 0.000001) == load2.reference.yversor
        assert pytest.approx(load1.reference.zversor, 0.000001) == load2.reference.zversor

        load2 = copy.deepcopy(load1)
        load2.rotate_along([1, 0, 0], 0)
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == load2.force
        assert pytest.approx(load1.moment, 0.000001) == load2.moment
        assert pytest.approx(load1.reference.origin, 0.000001) == load2.reference.origin
        assert pytest.approx(load1.reference.xversor, 0.000001) == load2.reference.xversor
        assert pytest.approx(load1.reference.yversor, 0.000001) == load2.reference.yversor
        assert pytest.approx(load1.reference.zversor, 0.000001) == load2.reference.zversor

        load2 = copy.deepcopy(load1)
        load2.rotate_along([0, 1, 0], 0)
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == load2.force
        assert pytest.approx(load1.moment, 0.000001) == load2.moment
        assert pytest.approx(load1.reference.origin, 0.000001) == load2.reference.origin
        assert pytest.approx(load1.reference.xversor, 0.000001) == load2.reference.xversor
        assert pytest.approx(load1.reference.yversor, 0.000001) == load2.reference.yversor
        assert pytest.approx(load1.reference.zversor, 0.000001) == load2.reference.zversor
    
        load2 = copy.deepcopy(load1)
        load2.rotate_along([0, 0, 1], 0)
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == load2.force
        assert pytest.approx(load1.moment, 0.000001) == load2.moment
        assert pytest.approx(load1.reference.origin, 0.000001) == load2.reference.origin
        assert pytest.approx(load1.reference.xversor, 0.000001) == load2.reference.xversor
        assert pytest.approx(load1.reference.yversor, 0.000001) == load2.reference.yversor
        assert pytest.approx(load1.reference.zversor, 0.000001) == load2.reference.zversor
    
        load2 = copy.deepcopy(load1)
        load2.rotate_along([1, 2, 3], 0)
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == load2.force
        assert pytest.approx(load1.moment, 0.000001) == load2.moment
        assert pytest.approx(load1.reference.origin, 0.000001) == load2.reference.origin
        assert pytest.approx(load1.reference.xversor, 0.000001) == load2.reference.xversor
        assert pytest.approx(load1.reference.yversor, 0.000001) == load2.reference.yversor
        assert pytest.approx(load1.reference.zversor, 0.000001) == load2.reference.zversor
    
        load2 = copy.deepcopy(load1)
        load2.rotate_along([-1, -1, -1], 0)
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == load2.force
        assert pytest.approx(load1.moment, 0.000001) == load2.moment
        assert pytest.approx(load1.reference.origin, 0.000001) == load2.reference.origin
        assert pytest.approx(load1.reference.xversor, 0.000001) == load2.reference.xversor
        assert pytest.approx(load1.reference.yversor, 0.000001) == load2.reference.yversor
        assert pytest.approx(load1.reference.zversor, 0.000001) == load2.reference.zversor
    
        load2 = copy.deepcopy(load1)
        load2.rotate_along([1, -1, -1], 0)
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == load2.force
        assert pytest.approx(load1.moment, 0.000001) == load2.moment
        assert pytest.approx(load1.reference.origin, 0.000001) == load2.reference.origin
        assert pytest.approx(load1.reference.xversor, 0.000001) == load2.reference.xversor
        assert pytest.approx(load1.reference.yversor, 0.000001) == load2.reference.yversor
        assert pytest.approx(load1.reference.zversor, 0.000001) == load2.reference.zversor
    
    # rotation arround x
    ref = ReferenceFrame()
    for theta in np.linspace(-180, 180, 18*4+1):
        cos = np.cos(np.radians(theta))
        sin = np.sin(np.radians(theta))
        force = [1, 0, 0]
        moment = [1, 0, 0]
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.rotate_along([1, 0, 0], theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == [1, 0, 0]
        assert pytest.approx(load1.moment, 0.000001) == [1, 0, 0]
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([1, 0, 0])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([0, cos, sin])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([0, -sin, cos])

        force = [0, 1, 0]
        moment = [0, 1, 0]
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.rotate_along([1, 0, 0], theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == [0, cos, -sin]
        assert pytest.approx(load1.moment, 0.000001) == [0, cos, -sin]
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([1, 0, 0])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([0, cos, sin])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([0, -sin, cos])


        force = [0, 0, 1]
        moment = [0, 0, 1]
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.rotate_along([1, 0, 0], theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == [0, sin, cos]
        assert pytest.approx(load1.moment, 0.000001) == [0, sin, cos]
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([1, 0, 0])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([0, cos, sin])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([0, -sin, cos])

        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.rotate_along([1, 0, 0], theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == np.array([force[0], force[1] * cos + force[2] * sin, - force[1] * sin + force[2] * cos])
        assert pytest.approx(load1.moment, 0.000001) == np.array([moment[0], moment[1] * cos + moment[2] * sin, -moment[1] * sin + moment[2] * cos])
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([1, 0, 0])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([0, cos, sin])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([0, -sin, cos])

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for theta in np.linspace(-180, 180, 18*4+1):
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            
            force_o = ref.r2o(force)
            moment_o = ref.r2o(moment)

            ref_r = copy.deepcopy(ref)
            ref_r.rotate_along([1, 0, 0], theta)
            load1.rotate_along([1, 0, 0], theta)

            force_r = ref_r.o2r(force_o)
            moment_r = ref_r.o2r(moment_o)

            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == force_r
            assert pytest.approx(load1.moment, 0.000001) == moment_r
            assert pytest.approx(load1.reference.origin, 0.000001) == ref_r.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == ref_r.xversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == ref_r.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == ref_r.zversor

    # rotation arround y
    ref = ReferenceFrame()
    for theta in np.linspace(-180, 180, 18*4+1):
        cos = np.cos(np.radians(theta))
        sin = np.sin(np.radians(theta))
        force = [0, 1, 0]
        moment = [0, 1, 0]
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.rotate_along([0, 1, 0], theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == [0, 1, 0]
        assert pytest.approx(load1.moment, 0.000001) == [0, 1, 0]
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([cos, 0, -sin])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([0, 1, 0])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([sin, 0, cos])

        force = [1, 0, 0]
        moment = [1, 0, 0]
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.rotate_along([0, 1, 0], theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == [cos, 0, sin]
        assert pytest.approx(load1.moment, 0.000001) == [cos, 0, sin]
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([cos, 0, -sin])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([0, 1, 0])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([sin, 0, cos])


        force = [0, 0, 1]
        moment = [0, 0, 1]
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.rotate_along([0, 1, 0], theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == [-sin, 0, cos]
        assert pytest.approx(load1.moment, 0.000001) == [-sin, 0, cos]
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([cos, 0, -sin])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([0, 1, 0])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([sin, 0, cos])

        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.rotate_along([0, 1, 0], theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == np.array([force[0] * cos - force[2] * sin, force[1], force[0] * sin + force[2] * cos])
        assert pytest.approx(load1.moment, 0.000001) == np.array([moment[0] * cos - moment[2] * sin, moment[1], moment[0] * sin + moment[2] * cos])
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([cos, 0, -sin])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([0, 1, 0])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([sin, 0, cos])
    
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for theta in np.linspace(-180, 180, 18*4+1):
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            
            force_o = ref.r2o(force)
            moment_o = ref.r2o(moment)

            ref_r = copy.deepcopy(ref)
            ref_r.rotate_along([0, 1, 0], theta)
            load1.rotate_along([0, 1, 0], theta)

            force_r = ref_r.o2r(force_o)
            moment_r = ref_r.o2r(moment_o)
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == force_r
            assert pytest.approx(load1.moment, 0.000001) == moment_r
            assert pytest.approx(load1.reference.origin, 0.000001) == ref_r.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == ref_r.xversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == ref_r.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == ref_r.zversor
    
    # rotation arround z
    ref = ReferenceFrame()
    for theta in np.linspace(-180, 180, 18*4+1):
        cos = np.cos(np.radians(theta))
        sin = np.sin(np.radians(theta))
        force = [0, 0, 1]
        moment = [0, 0, 1]
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.rotate_along([0, 0, 1], theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == [0, 0, 1]
        assert pytest.approx(load1.moment, 0.000001) == [0, 0, 1]
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([cos, sin, 0])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([-sin, cos, 0])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([0, 0, 1])

        force = [1, 0, 0]
        moment = [1, 0, 0]
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.rotate_along([0, 0, 1], theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == [cos, -sin, 0]
        assert pytest.approx(load1.moment, 0.000001) == [cos, -sin, 0 ]
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([cos, sin, 0])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([-sin, cos, 0])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([0, 0, 1])


        force = [0, 1, 0]
        moment = [0, 1, 0]
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.rotate_along([0, 0, 1], theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == [sin, cos, 0]
        assert pytest.approx(load1.moment, 0.000001) == [sin, cos, 0]
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([cos, sin, 0])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([-sin, cos, 0])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([0, 0, 1])

        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.rotate_along([0, 0, 1], theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == np.array([force[0] * cos + force[1] * sin, -force[0] * sin + force[1] * cos ,force[2]])
        assert pytest.approx(load1.moment, 0.000001) == np.array([moment[0] * cos + moment[1] * sin, -moment[0] * sin + moment[1] * cos ,moment[2]])
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([cos, sin, 0])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([-sin, cos, 0])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([0, 0, 1])
    
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for theta in np.linspace(-180, 180, 18*4+1):
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            
            force_o = ref.r2o(force)
            moment_o = ref.r2o(moment)

            ref_r = copy.deepcopy(ref)
            ref_r.rotate_along([0, 0, 1], theta)
            load1.rotate_along([0, 0, 1], theta)

            force_r = ref_r.o2r(force_o)
            moment_r = ref_r.o2r(moment_o)

            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == force_r
            assert pytest.approx(load1.moment, 0.000001) == moment_r
            assert pytest.approx(load1.reference.origin, 0.000001) == ref_r.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == ref_r.xversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == ref_r.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == ref_r.zversor
    
    # general rotation
    theta = np.linspace(-180, 180, 18+1)
    vals = [-123, 0, 3.5]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        ref = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        for lat, lon in itertools.product(theta, theta):                              
            direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                                  np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                                  np.sin(np.radians(lat))])
            for rot in theta:
                force = np.random.random(3) * 200 - 100
                moment = np.random.random(3) * 200 - 100
                load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
                
                force_o = ref.r2o(force)
                moment_o = ref.r2o(moment)

                ref_r = copy.deepcopy(ref)
                ref_r.rotate_along(direction, rot)
                load1.rotate_along(direction, rot)

                force_r = ref_r.o2r(force_o)
                moment_r = ref_r.o2r(moment_o)

                assert load1.loadtype == 'D'
                assert load1.name == 'test'
                assert pytest.approx(load1.force, 0.000001) == force_r
                assert pytest.approx(load1.moment, 0.000001) == moment_r
                assert pytest.approx(load1.reference.origin, 0.000001) == ref_r.origin
                assert pytest.approx(load1.reference.xversor, 0.000001) == ref_r.xversor
                assert pytest.approx(load1.reference.yversor, 0.000001) == ref_r.yversor
                assert pytest.approx(load1.reference.zversor, 0.000001) == ref_r.zversor
                
    
def test_xrotate():
    '''Test xrotate method from Load class
    '''

    # no shift
    load1 = Load('D', name='test', xforce=2, zforce=4, ymoment=-1)
    load2 = copy.deepcopy(load1)
    load2.xrotate(0)
    assert load1 == load2


    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        load1 = Load.fromarray('D', name='test', force=np.random.random(3),
                                      moment=np.random.random(3), reference=ref)
        load2 = copy.deepcopy(load1)
        load2.xrotate(0)
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == load2.force
        assert pytest.approx(load1.moment, 0.000001) == load2.moment
        assert pytest.approx(load1.reference.origin, 0.000001) == load2.reference.origin
        assert pytest.approx(load1.reference.xversor, 0.000001) == load2.reference.xversor
        assert pytest.approx(load1.reference.yversor, 0.000001) == load2.reference.yversor
        assert pytest.approx(load1.reference.zversor, 0.000001) == load2.reference.zversor
    
    # rotation arround x
    ref = ReferenceFrame()
    for theta in np.linspace(-180, 180, 18*4+1):
        cos = np.cos(np.radians(theta))
        sin = np.sin(np.radians(theta))
        force = [1, 0, 0]
        moment = [1, 0, 0]
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.xrotate(theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == [1, 0, 0]
        assert pytest.approx(load1.moment, 0.000001) == [1, 0, 0]
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([1, 0, 0])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([0, cos, sin])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([0, -sin, cos])

        force = [0, 1, 0]
        moment = [0, 1, 0]
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.xrotate(theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == [0, cos, -sin]
        assert pytest.approx(load1.moment, 0.000001) == [0, cos, -sin]
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([1, 0, 0])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([0, cos, sin])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([0, -sin, cos])


        force = [0, 0, 1]
        moment = [0, 0, 1]
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.xrotate(theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == [0, sin, cos]
        assert pytest.approx(load1.moment, 0.000001) == [0, sin, cos]
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([1, 0, 0])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([0, cos, sin])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([0, -sin, cos])

        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.xrotate(theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == np.array([force[0], force[1] * cos + force[2] * sin, - force[1] * sin + force[2] * cos])
        assert pytest.approx(load1.moment, 0.000001) == np.array([moment[0], moment[1] * cos + moment[2] * sin, -moment[1] * sin + moment[2] * cos])
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([1, 0, 0])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([0, cos, sin])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([0, -sin, cos])

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for theta in np.linspace(-180, 180, 18*4+1):
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            
            force_o = ref.r2o(force)
            moment_o = ref.r2o(moment)

            ref_r = copy.deepcopy(ref)
            ref_r.xrotate(theta)
            load1.xrotate(theta)

            force_r = ref_r.o2r(force_o)
            moment_r = ref_r.o2r(moment_o)
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == force_r
            assert pytest.approx(load1.moment, 0.000001) == moment_r
            assert pytest.approx(load1.reference.origin, 0.000001) == ref_r.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == ref_r.xversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == ref_r.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == ref_r.zversor


def test_yrotate():
    '''Test yrotate method from Load class
    '''

    # no shift
    load1 = Load('D', name='test', xforce=2, zforce=4, ymoment=-1)
    load2 = copy.deepcopy(load1)
    load2.yrotate(0)
    assert load1 == load2

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        load1 = Load.fromarray('D', name='test', force=np.random.random(3),
                                      moment=np.random.random(3), reference=ref)
        load2 = copy.deepcopy(load1)
        load2.yrotate(0)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == load2.force
        assert pytest.approx(load1.moment, 0.000001) == load2.moment
        assert pytest.approx(load1.reference.origin, 0.000001) == load2.reference.origin
        assert pytest.approx(load1.reference.xversor, 0.000001) == load2.reference.xversor
        assert pytest.approx(load1.reference.yversor, 0.000001) == load2.reference.yversor
        assert pytest.approx(load1.reference.zversor, 0.000001) == load2.reference.zversor


    ref = ReferenceFrame()
    for theta in np.linspace(-180, 180, 18*4+1):
        cos = np.cos(np.radians(theta))
        sin = np.sin(np.radians(theta))
        force = [0, 1, 0]
        moment = [0, 1, 0]
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.yrotate(theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == [0, 1, 0]
        assert pytest.approx(load1.moment, 0.000001) == [0, 1, 0]
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([cos, 0, -sin])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([0, 1, 0])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([sin, 0, cos])

        force = [1, 0, 0]
        moment = [1, 0, 0]
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.yrotate(theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == [cos, 0, sin]
        assert pytest.approx(load1.moment, 0.000001) == [cos, 0, sin]
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([cos, 0, -sin])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([0, 1, 0])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([sin, 0, cos])


        force = [0, 0, 1]
        moment = [0, 0, 1]
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.yrotate(theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == [-sin, 0, cos]
        assert pytest.approx(load1.moment, 0.000001) == [-sin, 0, cos]
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([cos, 0, -sin])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([0, 1, 0])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([sin, 0, cos])

        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.yrotate(theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == np.array([force[0] * cos - force[2] * sin, force[1], force[0] * sin + force[2] * cos])
        assert pytest.approx(load1.moment, 0.000001) == np.array([moment[0] * cos - moment[2] * sin, moment[1], moment[0] * sin + moment[2] * cos])
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([cos, 0, -sin])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([0, 1, 0])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([sin, 0, cos])
    
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for theta in np.linspace(-180, 180, 18*4+1):
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            
            force_o = ref.r2o(force)
            moment_o = ref.r2o(moment)

            ref_r = copy.deepcopy(ref)
            ref_r.yrotate(theta)
            load1.yrotate(theta)

            force_r = ref_r.o2r(force_o)
            moment_r = ref_r.o2r(moment_o)
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == force_r
            assert pytest.approx(load1.moment, 0.000001) == moment_r
            assert pytest.approx(load1.reference.origin, 0.000001) == ref_r.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == ref_r.xversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == ref_r.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == ref_r.zversor
    

def test_zrotate():
    '''Test zrotate method from Load class
    '''

    # no shift
    load1 = Load('D', name='test', xforce=2, zforce=4, ymoment=-1)
    load2 = copy.deepcopy(load1)
    load2.zrotate(0)
    assert load1 == load2

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        load1 = Load.fromarray('D', name='test', force=np.random.random(3),
                                      moment=np.random.random(3), reference=ref)
        load2 = copy.deepcopy(load1)
        load2.zrotate(0)
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == load2.force
        assert pytest.approx(load1.moment, 0.000001) == load2.moment
        assert pytest.approx(load1.reference.origin, 0.000001) == load2.reference.origin
        assert pytest.approx(load1.reference.xversor, 0.000001) == load2.reference.xversor
        assert pytest.approx(load1.reference.yversor, 0.000001) == load2.reference.yversor
        assert pytest.approx(load1.reference.zversor, 0.000001) == load2.reference.zversor

    ref = ReferenceFrame()
    for theta in np.linspace(-180, 180, 18*4+1):
        cos = np.cos(np.radians(theta))
        sin = np.sin(np.radians(theta))
        force = [0, 0, 1]
        moment = [0, 0, 1]
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.zrotate(theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == [0, 0, 1]
        assert pytest.approx(load1.moment, 0.000001) == [0, 0, 1]
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([cos, sin, 0])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([-sin, cos, 0])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([0, 0, 1])

        force = [1, 0, 0]
        moment = [1, 0, 0]
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.zrotate(theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == [cos, -sin, 0]
        assert pytest.approx(load1.moment, 0.000001) == [cos, -sin, 0 ]
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([cos, sin, 0])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([-sin, cos, 0])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([0, 0, 1])


        force = [0, 1, 0]
        moment = [0, 1, 0]
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.zrotate(theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == [sin, cos, 0]
        assert pytest.approx(load1.moment, 0.000001) == [sin, cos, 0]
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([cos, sin, 0])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([-sin, cos, 0])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([0, 0, 1])

        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load1.zrotate(theta)
        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == np.array([force[0] * cos + force[1] * sin, -force[0] * sin + force[1] * cos ,force[2]])
        assert pytest.approx(load1.moment, 0.000001) == np.array([moment[0] * cos + moment[1] * sin, -moment[0] * sin + moment[1] * cos ,moment[2]])
        assert pytest.approx(load1.reference.origin, 0.000001) == np.array([0, 0, 0])
        assert pytest.approx(load1.reference.xversor, 0.000001) == np.array([cos, sin, 0])
        assert pytest.approx(load1.reference.yversor, 0.000001) == np.array([-sin, cos, 0])
        assert pytest.approx(load1.reference.zversor, 0.000001) == np.array([0, 0, 1])
    
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for theta in np.linspace(-180, 180, 18*4+1):
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            
            force_o = ref.r2o(force)
            moment_o = ref.r2o(moment)

            ref_r = copy.deepcopy(ref)
            ref_r.zrotate(theta)
            load1.zrotate(theta)

            force_r = ref_r.o2r(force_o)
            moment_r = ref_r.o2r(moment_o)

            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == force_r
            assert pytest.approx(load1.moment, 0.000001) == moment_r
            assert pytest.approx(load1.reference.origin, 0.000001) == ref_r.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == ref_r.xversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == ref_r.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == ref_r.zversor


def test_rotate_along_ref():
    '''Test rotate_along_ref method from Load class
    '''
    # no shift
    load1 = Load('D', name='test', xforce=2, zforce=4, ymoment=-1)
    load2 = copy.deepcopy(load1)
    load2.rotate_along_ref([1, 1, 1], 0)
    assert load1 == load2
    load2 = copy.deepcopy(load1)
    load2.rotate_along_ref([1, 0, 0], 0)
    assert load1 == load2
    load2 = copy.deepcopy(load1)
    load2.rotate_along_ref([0, 1, 0], 0)
    assert load1 == load2
    load2 = copy.deepcopy(load1)
    load2.rotate_along_ref([0, 0, 1], 0)
    assert load1 == load2
    load2 = copy.deepcopy(load1)
    load2.rotate_along_ref([1, 2, 3], 0)
    assert load1 == load2
    load2 = copy.deepcopy(load1)
    load2.rotate_along_ref([-1, -1, -1], 0)
    assert load1 == load2
    load2 = copy.deepcopy(load1)
    load2.rotate_along_ref([1, -1, -1], 0)
    assert load1 == load2


    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        load1 = Load.fromarray('D', name='test', force=np.random.random(3),
                                      moment=np.random.random(3), reference=ref)
        load2 = copy.deepcopy(load1)
        load2.rotate_along_ref([1, 1, 1], 0)
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == load2.force
        assert pytest.approx(load1.moment, 0.000001) == load2.moment
        assert pytest.approx(load1.reference.origin, 0.000001) == load2.reference.origin
        assert pytest.approx(load1.reference.xversor, 0.000001) == load2.reference.xversor
        assert pytest.approx(load1.reference.yversor, 0.000001) == load2.reference.yversor
        assert pytest.approx(load1.reference.zversor, 0.000001) == load2.reference.zversor

        load2 = copy.deepcopy(load1)
        load2.rotate_along_ref([1, 0, 0], 0)
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == load2.force
        assert pytest.approx(load1.moment, 0.000001) == load2.moment
        assert pytest.approx(load1.reference.origin, 0.000001) == load2.reference.origin
        assert pytest.approx(load1.reference.xversor, 0.000001) == load2.reference.xversor
        assert pytest.approx(load1.reference.yversor, 0.000001) == load2.reference.yversor
        assert pytest.approx(load1.reference.zversor, 0.000001) == load2.reference.zversor

        load2 = copy.deepcopy(load1)
        load2.rotate_along_ref([0, 1, 0], 0)
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == load2.force
        assert pytest.approx(load1.moment, 0.000001) == load2.moment
        assert pytest.approx(load1.reference.origin, 0.000001) == load2.reference.origin
        assert pytest.approx(load1.reference.xversor, 0.000001) == load2.reference.xversor
        assert pytest.approx(load1.reference.yversor, 0.000001) == load2.reference.yversor
        assert pytest.approx(load1.reference.zversor, 0.000001) == load2.reference.zversor
    
        load2 = copy.deepcopy(load1)
        load2.rotate_along_ref([0, 0, 1], 0)
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == load2.force
        assert pytest.approx(load1.moment, 0.000001) == load2.moment
        assert pytest.approx(load1.reference.origin, 0.000001) == load2.reference.origin
        assert pytest.approx(load1.reference.xversor, 0.000001) == load2.reference.xversor
        assert pytest.approx(load1.reference.yversor, 0.000001) == load2.reference.yversor
        assert pytest.approx(load1.reference.zversor, 0.000001) == load2.reference.zversor
    
        load2 = copy.deepcopy(load1)
        load2.rotate_along_ref([1, 2, 3], 0)
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == load2.force
        assert pytest.approx(load1.moment, 0.000001) == load2.moment
        assert pytest.approx(load1.reference.origin, 0.000001) == load2.reference.origin
        assert pytest.approx(load1.reference.xversor, 0.000001) == load2.reference.xversor
        assert pytest.approx(load1.reference.yversor, 0.000001) == load2.reference.yversor
        assert pytest.approx(load1.reference.zversor, 0.000001) == load2.reference.zversor
    
        load2 = copy.deepcopy(load1)
        load2.rotate_along_ref([-1, -1, -1], 0)
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == load2.force
        assert pytest.approx(load1.moment, 0.000001) == load2.moment
        assert pytest.approx(load1.reference.origin, 0.000001) == load2.reference.origin
        assert pytest.approx(load1.reference.xversor, 0.000001) == load2.reference.xversor
        assert pytest.approx(load1.reference.yversor, 0.000001) == load2.reference.yversor
        assert pytest.approx(load1.reference.zversor, 0.000001) == load2.reference.zversor
    
        load2 = copy.deepcopy(load1)
        load2.rotate_along_ref([1, -1, -1], 0)
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == load2.force
        assert pytest.approx(load1.moment, 0.000001) == load2.moment
        assert pytest.approx(load1.reference.origin, 0.000001) == load2.reference.origin
        assert pytest.approx(load1.reference.xversor, 0.000001) == load2.reference.xversor
        assert pytest.approx(load1.reference.yversor, 0.000001) == load2.reference.yversor
        assert pytest.approx(load1.reference.zversor, 0.000001) == load2.reference.zversor

    # rotation arround x
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for theta in np.linspace(-180, 180, 18*4+1):
            cos = np.cos(np.radians(theta))
            sin = np.sin(np.radians(theta))
            force = [1, 0, 0]
            moment = [1, 0, 0]
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.rotate_along_ref([1, 0, 0], theta)
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == [1, 0, 0]
            assert pytest.approx(load1.moment, 0.000001) == [1, 0, 0]
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == ref.xversor
            assert pytest.approx(load1.reference.yversor, 0.000001) ==  cos * ref.yversor + sin * ref.zversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == - sin *  ref.yversor + cos * ref.zversor

            force = [0, 1, 0]
            moment = [0, 1, 0]
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.rotate_along_ref([1, 0, 0], theta)
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == [0, cos, -sin]
            assert pytest.approx(load1.moment, 0.000001) == [0, cos, -sin]
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == ref.xversor
            assert pytest.approx(load1.reference.yversor, 0.000001) ==  cos * ref.yversor + sin * ref.zversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == - sin *  ref.yversor + cos * ref.zversor


            force = [0, 0, 1]
            moment = [0, 0, 1]
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.rotate_along_ref([1, 0, 0], theta)
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == [0, sin, cos]
            assert pytest.approx(load1.moment, 0.000001) == [0, sin, cos]
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == ref.xversor
            assert pytest.approx(load1.reference.yversor, 0.000001) ==  cos * ref.yversor + sin * ref.zversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == - sin *  ref.yversor + cos * ref.zversor

            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.rotate_along_ref([1, 0, 0], theta)
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == np.array([force[0], force[1] * cos + force[2] * sin, - force[1] * sin + force[2] * cos])
            assert pytest.approx(load1.moment, 0.000001) == np.array([moment[0], moment[1] * cos + moment[2] * sin, -moment[1] * sin + moment[2] * cos])
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == ref.xversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == cos * ref.yversor + sin * ref.zversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == - sin *  ref.yversor + cos * ref.zversor
    
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for theta in np.linspace(-180, 180, 18*4+1):
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            
            force_o = ref.r2o(force)
            moment_o = ref.r2o(moment)

            ref_r = copy.deepcopy(ref)
            ref_r.rotate_along_ref([0, 1, 0], theta)
            load1.rotate_along_ref([0, 1, 0], theta)

            force_r = ref_r.o2r(force_o)
            moment_r = ref_r.o2r(moment_o)
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == force_r
            assert pytest.approx(load1.moment, 0.000001) == moment_r
            assert pytest.approx(load1.reference.origin, 0.000001) == ref_r.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == ref_r.xversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == ref_r.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == ref_r.zversor
    
    # rotation arround y
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for theta in np.linspace(-180, 180, 18*4+1):
            cos = np.cos(np.radians(theta))
            sin = np.sin(np.radians(theta))
            force = [0, 1, 0]
            moment = [0, 1, 0]
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.rotate_along_ref([0, 1, 0], theta)
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == [0, 1, 0]
            assert pytest.approx(load1.moment, 0.000001) == [0, 1, 0]
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == cos * ref.xversor - sin * ref.zversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == ref.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == sin * ref.xversor + cos * ref.zversor

            force = [1, 0, 0]
            moment = [1, 0, 0]
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.rotate_along_ref([0, 1, 0], theta)
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == [cos, 0, sin]
            assert pytest.approx(load1.moment, 0.000001) == [cos, 0, sin]
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == cos * ref.xversor - sin * ref.zversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == ref.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == sin * ref.xversor + cos * ref.zversor


            force = [0, 0, 1]
            moment = [0, 0, 1]
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.rotate_along_ref([0, 1, 0], theta)
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == [-sin, 0, cos]
            assert pytest.approx(load1.moment, 0.000001) == [-sin, 0, cos]
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == cos * ref.xversor - sin * ref.zversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == ref.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == sin * ref.xversor + cos * ref.zversor

            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.rotate_along_ref([0, 1, 0], theta)
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == np.array([force[0] * cos - force[2] * sin, force[1], force[0] * sin + force[2] * cos])
            assert pytest.approx(load1.moment, 0.000001) == np.array([moment[0] * cos - moment[2] * sin, moment[1], moment[0] * sin + moment[2] * cos])
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == cos * ref.xversor - sin * ref.zversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == ref.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == sin * ref.xversor + cos * ref.zversor
        
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for theta in np.linspace(-180, 180, 18*4+1):
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            
            force_o = ref.r2o(force)
            moment_o = ref.r2o(moment)

            ref_r = copy.deepcopy(ref)
            ref_r.rotate_along_ref([0, 1, 0], theta)
            load1.rotate_along_ref([0, 1, 0], theta)

            force_r = ref_r.o2r(force_o)
            moment_r = ref_r.o2r(moment_o)

            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == force_r
            assert pytest.approx(load1.moment, 0.000001) == moment_r
            assert pytest.approx(load1.reference.origin, 0.000001) == ref_r.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == ref_r.xversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == ref_r.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == ref_r.zversor

    # rotation arround z
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for theta in np.linspace(-180, 180, 18*4+1):
            cos = np.cos(np.radians(theta))
            sin = np.sin(np.radians(theta))
            force = [0, 0, 1]
            moment = [0, 0, 1]
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.rotate_along_ref([0, 0, 1], theta)
            
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == [0, 0, 1]
            assert pytest.approx(load1.moment, 0.000001) == [0, 0, 1]
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == cos * ref.xversor +  sin * ref.yversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == -sin * ref.xversor +  cos * ref.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == ref.zversor

            force = [1, 0, 0]
            moment = [1, 0, 0]
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.rotate_along_ref([0, 0, 1], theta)
            
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == [cos, -sin, 0]
            assert pytest.approx(load1.moment, 0.000001) == [cos, -sin, 0 ]
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == cos * ref.xversor +  sin * ref.yversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == -sin * ref.xversor +  cos * ref.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == ref.zversor


            force = [0, 1, 0]
            moment = [0, 1, 0]
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.rotate_along_ref([0, 0, 1], theta)
            
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == [sin, cos, 0]
            assert pytest.approx(load1.moment, 0.000001) == [sin, cos, 0]
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == cos * ref.xversor +  sin * ref.yversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == -sin * ref.xversor +  cos * ref.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == ref.zversor

            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.rotate_along_ref([0, 0, 1], theta)
            
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == np.array([force[0] * cos + force[1] * sin, -force[0] * sin + force[1] * cos ,force[2]])
            assert pytest.approx(load1.moment, 0.000001) == np.array([moment[0] * cos + moment[1] * sin, -moment[0] * sin + moment[1] * cos ,moment[2]])
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == cos * ref.xversor +  sin * ref.yversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == -sin * ref.xversor +  cos * ref.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == ref.zversor
    
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for theta in np.linspace(-180, 180, 18*4+1):
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            
            force_o = ref.r2o(force)
            moment_o = ref.r2o(moment)

            ref_r = copy.deepcopy(ref)
            ref_r.rotate_along_ref([0, 0, 1], theta)
            load1.rotate_along_ref([0, 0, 1], theta)

            force_r = ref_r.o2r(force_o)
            moment_r = ref_r.o2r(moment_o)

            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == force_r
            assert pytest.approx(load1.moment, 0.000001) == moment_r
            assert pytest.approx(load1.reference.origin, 0.000001) == ref_r.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == ref_r.xversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == ref_r.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == ref_r.zversor
    
    # general rotation
    theta = np.linspace(-180, 180, 18+1)
    vals = [-123, 0, 3.5]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        ref = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        for lat, lon in itertools.product(theta, theta):                              
            direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                                  np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                                  np.sin(np.radians(lat))])
            for rot in theta:
                force = np.random.random(3) * 200 - 100
                moment = np.random.random(3) * 200 - 100
                load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
                
                force_o = ref.r2o(force)
                moment_o = ref.r2o(moment)

                ref_r = copy.deepcopy(ref)
                ref_r.rotate_along_ref(direction, rot)
                load1.rotate_along_ref(direction, rot)

                force_r = ref_r.o2r(force_o)
                moment_r = ref_r.o2r(moment_o)

                assert load1.loadtype == 'D'
                assert load1.name == 'test'
                assert pytest.approx(load1.force, 0.000001) == force_r
                assert pytest.approx(load1.moment, 0.000001) == moment_r
                assert pytest.approx(load1.reference.origin, 0.000001) == ref_r.origin
                assert pytest.approx(load1.reference.xversor, 0.000001) == ref_r.xversor
                assert pytest.approx(load1.reference.yversor, 0.000001) == ref_r.yversor
                assert pytest.approx(load1.reference.zversor, 0.000001) == ref_r.zversor


def test_xrotate_ref():
    '''Test xrotate_ref method from Load class
    '''
    # no shift
    load1 = Load('D', name='test', xforce=2, zforce=4, ymoment=-1)
    load2 = copy.deepcopy(load1)
    load2.xrotate_ref(0)
    assert load1 == load2


    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        load1 = Load.fromarray('D', name='test', force=np.random.random(3),
                                      moment=np.random.random(3), reference=ref)
        load2 = copy.deepcopy(load1)
        load2.xrotate_ref(0)

        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == load2.force
        assert pytest.approx(load1.moment, 0.000001) == load2.moment
        assert pytest.approx(load1.reference.origin, 0.000001) == load2.reference.origin
        assert pytest.approx(load1.reference.xversor, 0.000001) == load2.reference.xversor
        assert pytest.approx(load1.reference.yversor, 0.000001) == load2.reference.yversor
        assert pytest.approx(load1.reference.zversor, 0.000001) == load2.reference.zversor

    # rotation arround x
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for theta in np.linspace(-180, 180, 18*4+1):
            cos = np.cos(np.radians(theta))
            sin = np.sin(np.radians(theta))
            force = [1, 0, 0]
            moment = [1, 0, 0]
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.xrotate_ref(theta)

            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == [1, 0, 0]
            assert pytest.approx(load1.moment, 0.000001) == [1, 0, 0]
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == ref.xversor
            assert pytest.approx(load1.reference.yversor, 0.000001) ==  cos * ref.yversor + sin * ref.zversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == - sin *  ref.yversor + cos * ref.zversor

            force = [0, 1, 0]
            moment = [0, 1, 0]
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.xrotate_ref(theta)

            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == [0, cos, -sin]
            assert pytest.approx(load1.moment, 0.000001) == [0, cos, -sin]
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == ref.xversor
            assert pytest.approx(load1.reference.yversor, 0.000001) ==  cos * ref.yversor + sin * ref.zversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == - sin *  ref.yversor + cos * ref.zversor


            force = [0, 0, 1]
            moment = [0, 0, 1]
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.xrotate_ref(theta)

            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == [0, sin, cos]
            assert pytest.approx(load1.moment, 0.000001) == [0, sin, cos]
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == ref.xversor
            assert pytest.approx(load1.reference.yversor, 0.000001) ==  cos * ref.yversor + sin * ref.zversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == - sin *  ref.yversor + cos * ref.zversor

            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.xrotate_ref(theta)

            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == np.array([force[0], force[1] * cos + force[2] * sin, - force[1] * sin + force[2] * cos])
            assert pytest.approx(load1.moment, 0.000001) == np.array([moment[0], moment[1] * cos + moment[2] * sin, -moment[1] * sin + moment[2] * cos])
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == ref.xversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == cos * ref.yversor + sin * ref.zversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == - sin *  ref.yversor + cos * ref.zversor
  

def test_yrotate_ref():
    '''Test yrotate_ref method from Load class
    '''
    # no shift
    load1 = Load('D', name='test', xforce=2, zforce=4, ymoment=-1)
    load2 = copy.deepcopy(load1)
    load2.yrotate_ref(0)
    assert load1 == load2


    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        load1 = Load.fromarray('D', name='test', force=np.random.random(3),
                                      moment=np.random.random(3), reference=ref)
        load2 = copy.deepcopy(load1)
        load2.yrotate_ref(0)

        assert load1.loadtype == 'D'
        assert load1.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == load2.force
        assert pytest.approx(load1.moment, 0.000001) == load2.moment
        assert pytest.approx(load1.reference.origin, 0.000001) == load2.reference.origin
        assert pytest.approx(load1.reference.xversor, 0.000001) == load2.reference.xversor
        assert pytest.approx(load1.reference.yversor, 0.000001) == load2.reference.yversor
        assert pytest.approx(load1.reference.zversor, 0.000001) == load2.reference.zversor

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for theta in np.linspace(-180, 180, 18*4+1):
            cos = np.cos(np.radians(theta))
            sin = np.sin(np.radians(theta))
            force = [0, 1, 0]
            moment = [0, 1, 0]
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.yrotate_ref(theta)

            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == [0, 1, 0]
            assert pytest.approx(load1.moment, 0.000001) == [0, 1, 0]
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == cos * ref.xversor - sin * ref.zversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == ref.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == sin * ref.xversor + cos * ref.zversor

            force = [1, 0, 0]
            moment = [1, 0, 0]
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.yrotate_ref(theta)

            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == [cos, 0, sin]
            assert pytest.approx(load1.moment, 0.000001) == [cos, 0, sin]
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == cos * ref.xversor - sin * ref.zversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == ref.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == sin * ref.xversor + cos * ref.zversor


            force = [0, 0, 1]
            moment = [0, 0, 1]
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.yrotate_ref(theta)

            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == [-sin, 0, cos]
            assert pytest.approx(load1.moment, 0.000001) == [-sin, 0, cos]
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == cos * ref.xversor - sin * ref.zversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == ref.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == sin * ref.xversor + cos * ref.zversor

            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.yrotate_ref(theta)

            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == np.array([force[0] * cos - force[2] * sin, force[1], force[0] * sin + force[2] * cos])
            assert pytest.approx(load1.moment, 0.000001) == np.array([moment[0] * cos - moment[2] * sin, moment[1], moment[0] * sin + moment[2] * cos])
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == cos * ref.xversor - sin * ref.zversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == ref.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == sin * ref.xversor + cos * ref.zversor
        
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for theta in np.linspace(-180, 180, 18*4+1):
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            
            force_o = ref.r2o(force)
            moment_o = ref.r2o(moment)

            ref_r = copy.deepcopy(ref)
            ref_r.yrotate_ref(theta)
            load1.yrotate_ref(theta)

            force_r = ref_r.o2r(force_o)
            moment_r = ref_r.o2r(moment_o)

            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == force_r
            assert pytest.approx(load1.moment, 0.000001) == moment_r
            assert pytest.approx(load1.reference.origin, 0.000001) == ref_r.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == ref_r.xversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == ref_r.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == ref_r.zversor


def test_zrotate_ref():
    '''Test zrotate_ref method from Load class
    '''
    # no shift
    load1 = Load('D', name='test', xforce=2, zforce=4, ymoment=-1)
    load2 = copy.deepcopy(load1)
    load2.zrotate_ref(0)
    assert load1 == load2


    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        load1 = Load.fromarray('D', name='test', force=np.random.random(3),
                                      moment=np.random.random(3), reference=ref)
        load2 = copy.deepcopy(load1)
        load2.zrotate_ref(0)

        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert pytest.approx(load1.force, 0.000001) == load2.force
        assert pytest.approx(load1.moment, 0.000001) == load2.moment
        assert pytest.approx(load1.reference.origin, 0.000001) == load2.reference.origin
        assert pytest.approx(load1.reference.xversor, 0.000001) == load2.reference.xversor
        assert pytest.approx(load1.reference.yversor, 0.000001) == load2.reference.yversor
        assert pytest.approx(load1.reference.zversor, 0.000001) == load2.reference.zversor

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for theta in np.linspace(-180, 180, 18*4+1):
            cos = np.cos(np.radians(theta))
            sin = np.sin(np.radians(theta))
            force = [0, 0, 1]
            moment = [0, 0, 1]
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.zrotate_ref(theta)

            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == [0, 0, 1]
            assert pytest.approx(load1.moment, 0.000001) == [0, 0, 1]
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == cos * ref.xversor +  sin * ref.yversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == -sin * ref.xversor +  cos * ref.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == ref.zversor

            force = [1, 0, 0]
            moment = [1, 0, 0]
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.zrotate_ref(theta)
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == [cos, -sin, 0]
            assert pytest.approx(load1.moment, 0.000001) == [cos, -sin, 0 ]
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == cos * ref.xversor +  sin * ref.yversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == -sin * ref.xversor +  cos * ref.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == ref.zversor


            force = [0, 1, 0]
            moment = [0, 1, 0]
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.zrotate_ref(theta)
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == [sin, cos, 0]
            assert pytest.approx(load1.moment, 0.000001) == [sin, cos, 0]
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == cos * ref.xversor +  sin * ref.yversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == -sin * ref.xversor +  cos * ref.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == ref.zversor

            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load1.zrotate_ref(theta)
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == np.array([force[0] * cos + force[1] * sin, -force[0] * sin + force[1] * cos ,force[2]])
            assert pytest.approx(load1.moment, 0.000001) == np.array([moment[0] * cos + moment[1] * sin, -moment[0] * sin + moment[1] * cos ,moment[2]])
            assert pytest.approx(load1.reference.origin, 0.000001) == ref.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == cos * ref.xversor +  sin * ref.yversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == -sin * ref.xversor +  cos * ref.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == ref.zversor
    
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for theta in np.linspace(-180, 180, 18*4+1):
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load1 = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            
            force_o = ref.r2o(force)
            moment_o = ref.r2o(moment)

            ref_r = copy.deepcopy(ref)
            ref_r.zrotate_ref(theta)
            load1.zrotate_ref(theta)

            force_r = ref_r.o2r(force_o)
            moment_r = ref_r.o2r(moment_o)
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.000001) == force_r
            assert pytest.approx(load1.moment, 0.000001) == moment_r
            assert pytest.approx(load1.reference.origin, 0.000001) == ref_r.origin
            assert pytest.approx(load1.reference.xversor, 0.000001) == ref_r.xversor
            assert pytest.approx(load1.reference.yversor, 0.000001) == ref_r.yversor
            assert pytest.approx(load1.reference.zversor, 0.000001) == ref_r.zversor


def test_to_reference_shift():
    '''Test to_reference method from Load class for shifts
    '''

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for _ in range(100):
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load.to_reference(ref)

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force
            assert pytest.approx(load.moment, 0.0000001) == moment
            assert load.reference == ref
    
    # shift in x
    coord = [-84.2, -0.4, 0, 3, 18.2]
    shifts = [-23.5, -0.6, 0, 1.7, 1234]
    for x0, y0, z0 in itertools.product(coord, coord, coord):
        for xshift in shifts:
            ref = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            ref.xshift(xshift)
            load.to_reference(ref)

            moment[1] = moment[1] + xshift * force[2]
            moment[2] = moment[2] - xshift * force[1]

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force
            assert pytest.approx(load.moment, 0.0000001) == moment
            assert load.reference == ref
    
    # shift in y
    coord = [-84.2, -0.4, 0, 3, 18.2]
    shifts = [-23.5, -0.6, 0, 1.7, 1234]
    for x0, y0, z0 in itertools.product(coord, coord, coord):
        for yshift in shifts:
            ref = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            ref.yshift(yshift)
            load.to_reference(ref)

            moment[0] = moment[0] - yshift * force[2]
            moment[2] = moment[2] + yshift * force[0]

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force
            assert pytest.approx(load.moment, 0.0000001) == moment
            assert load.reference == ref

    # shift in z
    coord = [-84.2, -0.4, 0, 3, 18.2]
    shifts = [-23.5, -0.6, 0, 1.7, 1234]
    for x0, y0, z0 in itertools.product(coord, coord, coord):
        for zshift in shifts:
            ref = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            ref.zshift(zshift)
            load.to_reference(ref)

            moment[0] = moment[0] + zshift * force[1]
            moment[1] = moment[1] - zshift * force[0]
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force
            assert pytest.approx(load.moment, 0.0000001) == moment
            assert load.reference == ref

    # shift in x, y and z
    coord = [-84.2, -0.4, 0, 3, 18.2]
    shifts = [-23.5, -0.6, 0, 1.7, 1234]
    for x0, y0, z0 in itertools.product(coord, coord, coord):
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            ref.shift(xshift, yshift, zshift)
            load.to_reference(ref)

            moment[1] = moment[1] + xshift * force[2]
            moment[2] = moment[2] - xshift * force[1]

            moment[0] = moment[0] - yshift * force[2]
            moment[2] = moment[2] + yshift * force[0]

            moment[0] = moment[0] + zshift * force[1]
            moment[1] = moment[1] - zshift * force[0]
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force
            assert pytest.approx(load.moment, 0.0000001) == moment
            assert load.reference == ref


def test_to_reference_xrotation():
    '''Test to_reference method from Load class for x rotations
    '''
    
    # rotation arround x
    thetas = [-180, -134.5, -27, 0, 45, 90, 105.3]
    for theta in thetas:
        ref = ReferenceFrame()
        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        ref.xrotate(theta)
        load.to_reference(ref)

        force2 = copy.deepcopy(force)
        moment2 = copy.deepcopy(moment)
        cos = np.cos(np.radians(theta))
        sin = np.sin(np.radians(theta))
        force2[1] = cos * force[1] + sin * force[2]
        force2[2] = -sin * force[1] + cos * force[2]
        moment2[1] = cos * moment[1] + sin * moment[2]
        moment2[2] = -sin * moment[1] + cos * moment[2]
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.0000001) == force2
        assert pytest.approx(load.moment, 0.0000001) == moment2
        assert load.reference == ref

    for theta in thetas:
        ref = ReferenceFrame()
        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load2 = copy.deepcopy(load)

        ref.xrotate(theta)
        load.to_reference(ref)
        load2.xrotate(theta)

        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.0000001) == load2.force
        assert pytest.approx(load.moment, 0.0000001) == load2.moment
        assert load.reference == load2.reference
    
    for theta in thetas:
        ref = ReferenceFrame()
        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        ref.xrotate(theta)
        load.to_reference(ref)

        force2 = ref.o2r(force)
        moment2 = ref.o2r(moment)
        
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.0000001) == force2
        assert pytest.approx(load.moment, 0.0000001) == moment2
        assert load.reference == ref

    # rotation + displacement
    shifts = [-23.5, -0.6, 0, 1.7, 1234]
    thetas = [-180, -134.5, -27, 0, 45, 90, 105.3]
    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            ref.xrotate(theta)
            ref.shift(xshift, yshift, zshift)
            load.to_reference(ref)


            moment[1] = moment[1] + xshift * force[2]
            moment[2] = moment[2] - xshift * force[1]

            moment[0] = moment[0] - yshift * force[2]
            moment[2] = moment[2] + yshift * force[0]

            moment[0] = moment[0] + zshift * force[1]
            moment[1] = moment[1] - zshift * force[0]

            force2 = copy.deepcopy(force)
            moment2 = copy.deepcopy(moment)
            cos = np.cos(np.radians(theta))
            sin = np.sin(np.radians(theta))
            force2[1] = cos * force[1] + sin * force[2]
            force2[2] = -sin * force[1] + cos * force[2]
            moment2[1] = cos * moment[1] + sin * moment[2]
            moment2[2] = -sin * moment[1] + cos * moment[2]

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force2
            assert pytest.approx(load.moment, 0.0000001) == moment2
            assert load.reference == ref

    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load2 = copy.deepcopy(load)

            ref.xrotate(theta)
            ref.shift(xshift, yshift, zshift)
            load.to_reference(ref)
            load2.xrotate(theta)
            load2.shift(xshift, yshift, zshift)

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == load2.force
            assert pytest.approx(load.moment, 0.0000001) == load2.moment
            assert load.reference == load2.reference
    
    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            ref.xrotate(theta)
            ref.shift(xshift, yshift, zshift)
            load.to_reference(ref)

            moment[1] = moment[1] + xshift * force[2]
            moment[2] = moment[2] - xshift * force[1]

            moment[0] = moment[0] - yshift * force[2]
            moment[2] = moment[2] + yshift * force[0]

            moment[0] = moment[0] + zshift * force[1]
            moment[1] = moment[1] - zshift * force[0]

            force2 = ref.o2r(force)
            moment2 = ref.o2r(moment)
            
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force2
            assert pytest.approx(load.moment, 0.0000001) == moment2
            assert load.reference == ref
    
    # displacement + rotation
    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            
            ref.shift(xshift, yshift, zshift)
            load.to_reference(ref)
            ref.xrotate(theta)
            load.to_reference(ref)

            moment[1] = moment[1] + xshift * force[2]
            moment[2] = moment[2] - xshift * force[1]

            moment[0] = moment[0] - yshift * force[2]
            moment[2] = moment[2] + yshift * force[0]

            moment[0] = moment[0] + zshift * force[1]
            moment[1] = moment[1] - zshift * force[0]

            force2 = copy.deepcopy(force)
            moment2 = copy.deepcopy(moment)
            cos = np.cos(np.radians(theta))
            sin = np.sin(np.radians(theta))
            force2[1] = cos * force[1] + sin * force[2]
            force2[2] = -sin * force[1] + cos * force[2]
            moment2[1] = cos * moment[1] + sin * moment[2]
            moment2[2] = -sin * moment[1] + cos * moment[2]

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force2
            assert pytest.approx(load.moment, 0.0000001) == moment2
            assert load.reference == ref

    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load2 = copy.deepcopy(load)

            ref.shift(xshift, yshift, zshift)
            load.to_reference(ref)
            ref.xrotate(theta)
            load.to_reference(ref)

            load2.shift(xshift, yshift, zshift)
            load2.xrotate(theta)
            
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == load2.force
            assert pytest.approx(load.moment, 0.0000001) == load2.moment
            assert load.reference == load2.reference
    
    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            ref.shift(xshift, yshift, zshift)
            load.to_reference(ref)
            ref.xrotate(theta)
            load.to_reference(ref)

            moment[1] = moment[1] + xshift * force[2]
            moment[2] = moment[2] - xshift * force[1]

            moment[0] = moment[0] - yshift * force[2]
            moment[2] = moment[2] + yshift * force[0]

            moment[0] = moment[0] + zshift * force[1]
            moment[1] = moment[1] - zshift * force[0]

            force2 = ref.o2r(force)
            moment2 = ref.o2r(moment)
            
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force2
            assert pytest.approx(load.moment, 0.0000001) == moment2
            assert load.reference == ref
    

def test_to_reference_yrotation():
    '''Test to_reference method from Load class for y rotations
    '''

    # rotation arround y
    thetas = [-180, -134.5, -27, 0, 45, 90, 105.3]
    for theta in thetas:
        ref = ReferenceFrame()
        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        ref.yrotate(theta)
        load.to_reference(ref)

        force2 = copy.deepcopy(force)
        moment2 = copy.deepcopy(moment)
        cos = np.cos(np.radians(theta))
        sin = np.sin(np.radians(theta))
        force2[0] = cos * force[0] - sin * force[2]
        force2[2] = sin * force[0] + cos * force[2]
        moment2[0] = cos * moment[0] - sin * moment[2]
        moment2[2] = sin * moment[0] + cos * moment[2]

        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.0000001) == force2
        assert pytest.approx(load.moment, 0.0000001) == moment2
        assert load.reference == ref

    for theta in thetas:
        ref = ReferenceFrame()
        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load2 = copy.deepcopy(load)

        ref.yrotate(theta)
        load.to_reference(ref)
        load2.yrotate(theta)

        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.0000001) == load2.force
        assert pytest.approx(load.moment, 0.0000001) == load2.moment
        assert load.reference == load2.reference
    
    for theta in thetas:
        ref = ReferenceFrame()
        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        ref.yrotate(theta)
        load.to_reference(ref)

        force2 = ref.o2r(force)
        moment2 = ref.o2r(moment)
        
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.0000001) == force2
        assert pytest.approx(load.moment, 0.0000001) == moment2
        assert load.reference == ref

    # rotation + displacement
    shifts = [-23.5, -0.6, 0, 1.7, 1234]
    thetas = [-180, -134.5, -27, 0, 45, 90, 105.3]
    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            ref.yrotate(theta)
            ref.shift(xshift, yshift, zshift)
            load.to_reference(ref)


            moment[1] = moment[1] + xshift * force[2]
            moment[2] = moment[2] - xshift * force[1]

            moment[0] = moment[0] - yshift * force[2]
            moment[2] = moment[2] + yshift * force[0]

            moment[0] = moment[0] + zshift * force[1]
            moment[1] = moment[1] - zshift * force[0]

            force2 = copy.deepcopy(force)
            moment2 = copy.deepcopy(moment)
            cos = np.cos(np.radians(theta))
            sin = np.sin(np.radians(theta))
            force2[0] = cos * force[0] - sin * force[2]
            force2[2] = sin * force[0] + cos * force[2]
            moment2[0] = cos * moment[0] - sin * moment[2]
            moment2[2] = sin * moment[0] + cos * moment[2]

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force2
            assert pytest.approx(load.moment, 0.0000001) == moment2
            assert load.reference == ref

    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load2 = copy.deepcopy(load)

            ref.yrotate(theta)
            ref.shift(xshift, yshift, zshift)
            load.to_reference(ref)
            load2.yrotate(theta)
            load2.shift(xshift, yshift, zshift)

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == load2.force
            assert pytest.approx(load.moment, 0.0000001) == load2.moment
            assert load.reference == load2.reference
    
    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            ref.yrotate(theta)
            ref.shift(xshift, yshift, zshift)
            load.to_reference(ref)

            moment[1] = moment[1] + xshift * force[2]
            moment[2] = moment[2] - xshift * force[1]

            moment[0] = moment[0] - yshift * force[2]
            moment[2] = moment[2] + yshift * force[0]

            moment[0] = moment[0] + zshift * force[1]
            moment[1] = moment[1] - zshift * force[0]

            force2 = ref.o2r(force)
            moment2 = ref.o2r(moment)
            
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force2
            assert pytest.approx(load.moment, 0.0000001) == moment2
            assert load.reference == ref
    
    # displacement + rotation
    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            
            ref.shift(xshift, yshift, zshift)
            load.to_reference(ref)
            ref.yrotate(theta)
            load.to_reference(ref)

            moment[1] = moment[1] + xshift * force[2]
            moment[2] = moment[2] - xshift * force[1]

            moment[0] = moment[0] - yshift * force[2]
            moment[2] = moment[2] + yshift * force[0]

            moment[0] = moment[0] + zshift * force[1]
            moment[1] = moment[1] - zshift * force[0]

            force2 = copy.deepcopy(force)
            moment2 = copy.deepcopy(moment)
            cos = np.cos(np.radians(theta))
            sin = np.sin(np.radians(theta))
            force2[0] = cos * force[0] - sin * force[2]
            force2[2] = sin * force[0] + cos * force[2]
            moment2[0] = cos * moment[0] - sin * moment[2]
            moment2[2] = sin * moment[0] + cos * moment[2]

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force2
            assert pytest.approx(load.moment, 0.0000001) == moment2
            assert load.reference == ref

    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load2 = copy.deepcopy(load)

            ref.shift(xshift, yshift, zshift)
            load.to_reference(ref)
            ref.yrotate(theta)
            load.to_reference(ref)

            load2.shift(xshift, yshift, zshift)
            load2.yrotate(theta)
            
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == load2.force
            assert pytest.approx(load.moment, 0.0000001) == load2.moment
            assert load.reference == load2.reference
    
    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            ref.shift(xshift, yshift, zshift)
            load.to_reference(ref)
            ref.yrotate(theta)
            load.to_reference(ref)

            moment[1] = moment[1] + xshift * force[2]
            moment[2] = moment[2] - xshift * force[1]

            moment[0] = moment[0] - yshift * force[2]
            moment[2] = moment[2] + yshift * force[0]

            moment[0] = moment[0] + zshift * force[1]
            moment[1] = moment[1] - zshift * force[0]

            force2 = ref.o2r(force)
            moment2 = ref.o2r(moment)
            
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force2
            assert pytest.approx(load.moment, 0.0000001) == moment2
            assert load.reference == ref


def test_to_reference_zrotation():
    '''Test to_reference method from Load class for z rotations
    '''

    # rotation arround z
    thetas = [-180, -134.5, -27, 0, 45, 90, 105.3]
    for theta in thetas:
        ref = ReferenceFrame()
        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        ref.zrotate(theta)
        load.to_reference(ref)

        force2 = copy.deepcopy(force)
        moment2 = copy.deepcopy(moment)
        cos = np.cos(np.radians(theta))
        sin = np.sin(np.radians(theta))
        force2[0] = cos * force[0] + sin * force[1]
        force2[1] = -sin * force[0] + cos * force[1]
        moment2[0] = cos * moment[0] + sin * moment[1]
        moment2[1] = -sin * moment[0] + cos * moment[1]

        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.0000001) == force2
        assert pytest.approx(load.moment, 0.0000001) == moment2
        assert load.reference == ref

    for theta in thetas:
        ref = ReferenceFrame()
        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load2 = copy.deepcopy(load)

        ref.zrotate(theta)
        load.to_reference(ref)
        load2.zrotate(theta)

        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.0000001) == load2.force
        assert pytest.approx(load.moment, 0.0000001) == load2.moment
        assert load.reference == load2.reference
    
    for theta in thetas:
        ref = ReferenceFrame()
        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        ref.zrotate(theta)
        load.to_reference(ref)

        force2 = ref.o2r(force)
        moment2 = ref.o2r(moment)
        
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.0000001) == force2
        assert pytest.approx(load.moment, 0.0000001) == moment2
        assert load.reference == ref

    # rotation + displacement
    shifts = [-23.5, -0.6, 0, 1.7, 1234]
    thetas = [-180, -134.5, -27, 0, 45, 90, 105.3]
    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            ref.zrotate(theta)
            ref.shift(xshift, yshift, zshift)
            load.to_reference(ref)


            moment[1] = moment[1] + xshift * force[2]
            moment[2] = moment[2] - xshift * force[1]

            moment[0] = moment[0] - yshift * force[2]
            moment[2] = moment[2] + yshift * force[0]

            moment[0] = moment[0] + zshift * force[1]
            moment[1] = moment[1] - zshift * force[0]

            force2 = copy.deepcopy(force)
            moment2 = copy.deepcopy(moment)
            cos = np.cos(np.radians(theta))
            sin = np.sin(np.radians(theta))
            force2[0] = cos * force[0] + sin * force[1]
            force2[1] = -sin * force[0] + cos * force[1]
            moment2[0] = cos * moment[0] + sin * moment[1]
            moment2[1] = -sin * moment[0] + cos * moment[1]

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force2
            assert pytest.approx(load.moment, 0.0000001) == moment2
            assert load.reference == ref

    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load2 = copy.deepcopy(load)

            ref.zrotate(theta)
            ref.shift(xshift, yshift, zshift)
            load.to_reference(ref)
            load2.zrotate(theta)
            load2.shift(xshift, yshift, zshift)

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == load2.force
            assert pytest.approx(load.moment, 0.0000001) == load2.moment
            assert load.reference == load2.reference
    
    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            ref.zrotate(theta)
            ref.shift(xshift, yshift, zshift)
            load.to_reference(ref)

            moment[1] = moment[1] + xshift * force[2]
            moment[2] = moment[2] - xshift * force[1]

            moment[0] = moment[0] - yshift * force[2]
            moment[2] = moment[2] + yshift * force[0]

            moment[0] = moment[0] + zshift * force[1]
            moment[1] = moment[1] - zshift * force[0]

            force2 = ref.o2r(force)
            moment2 = ref.o2r(moment)
            
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force2
            assert pytest.approx(load.moment, 0.0000001) == moment2
            assert load.reference == ref
    
    # displacement + rotation
    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            
            ref.shift(xshift, yshift, zshift)
            load.to_reference(ref)
            ref.zrotate(theta)
            load.to_reference(ref)

            moment[1] = moment[1] + xshift * force[2]
            moment[2] = moment[2] - xshift * force[1]

            moment[0] = moment[0] - yshift * force[2]
            moment[2] = moment[2] + yshift * force[0]

            moment[0] = moment[0] + zshift * force[1]
            moment[1] = moment[1] - zshift * force[0]

            force2 = copy.deepcopy(force)
            moment2 = copy.deepcopy(moment)
            cos = np.cos(np.radians(theta))
            sin = np.sin(np.radians(theta))
            force2[0] = cos * force[0] + sin * force[1]
            force2[1] = -sin * force[0] + cos * force[1]
            moment2[0] = cos * moment[0] + sin * moment[1]
            moment2[1] = -sin * moment[0] + cos * moment[1]

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force2
            assert pytest.approx(load.moment, 0.0000001) == moment2
            assert load.reference == ref

    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load2 = copy.deepcopy(load)

            ref.shift(xshift, yshift, zshift)
            load.to_reference(ref)
            ref.zrotate(theta)
            load.to_reference(ref)

            load2.shift(xshift, yshift, zshift)
            load2.zrotate(theta)
            
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == load2.force
            assert pytest.approx(load.moment, 0.0000001) == load2.moment
            assert load.reference == load2.reference
    
    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            ref.shift(xshift, yshift, zshift)
            load.to_reference(ref)
            ref.zrotate(theta)
            load.to_reference(ref)

            moment[1] = moment[1] + xshift * force[2]
            moment[2] = moment[2] - xshift * force[1]

            moment[0] = moment[0] - yshift * force[2]
            moment[2] = moment[2] + yshift * force[0]

            moment[0] = moment[0] + zshift * force[1]
            moment[1] = moment[1] - zshift * force[0]

            force2 = ref.o2r(force)
            moment2 = ref.o2r(moment)
            
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force2
            assert pytest.approx(load.moment, 0.0000001) == moment2
            assert load.reference == ref


def test_to_reference_rotation():
    '''Test to_reference method from Load class for generic
    rotation
    '''

    # general rotation
    coord = [-84.2, -0.4, 0, 3, 18.2]
    thetas = [-180, -134.5, -27, 0, 45, 90, 105.3]
    for x0, y0, z0 in itertools.product(coord, coord, coord):
        for lat, lon, theta in itertools.product(thetas, thetas, thetas): 
            ref = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)

            direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                                  np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                                  np.sin(np.radians(lat))])
            ref2 = copy.deepcopy(ref)
            ref2.rotate_along(direction, theta)

            load.to_reference(ref2)

            force_o = ref.r2o(force)
            moment_o = ref.r2o(moment)

            force2 = ref2.o2r(force_o)
            moment2 = ref2.o2r(moment_o)

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force2
            assert pytest.approx(load.moment, 0.0000001) == moment2
            assert load.reference == ref2
    
    coord = [-84.2, -0.4, 0, 3, 18.2]
    thetas = [-180, -134.5, -27, 0, 45, 90, 105.3]
    for x0, y0, z0 in itertools.product(coord, coord, coord):
        for lat, lon, theta in itertools.product(thetas, thetas, thetas): 
            ref = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)

            direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                                  np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                                  np.sin(np.radians(lat))])
            ref2 = copy.deepcopy(ref)
            ref2.rotate_along(direction, theta)

            load2 = copy.deepcopy(load)
            load.to_reference(ref2)

            load2.rotate_along(direction, theta)

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == load2.force
            assert pytest.approx(load.moment, 0.0000001) == load2.moment
            assert load.reference == load2.reference


def test_to_reference_rotation_shift():
    '''Test to_reference method from Load class for rotation +
    shifts
    '''

    # general rotation
    coord = [-84.2, 0, 3]
    thetas = [-180, -134.5, 0, 45, 105.3]
    shifts = [-23.5, 0, 1.7]
    for x0, y0, z0 in itertools.product(coord, coord, coord):
        for lat, lon, theta in itertools.product(thetas, thetas, thetas):
            for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
                ref = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                force = np.random.random(3) * 200 - 100
                moment = np.random.random(3) * 200 - 100
                load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)

                direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                                    np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                                    np.sin(np.radians(lat))])
                ref2 = copy.deepcopy(ref)
                ref2.rotate_along(direction, theta)
                ref2.shift(xshift, yshift, zshift)
                load.to_reference(ref2)

                force_o = ref.r2o(force)
                moment_o = ref.r2o(moment)

                moment_o[1] = moment_o[1] + xshift * force_o[2]
                moment_o[2] = moment_o[2] - xshift * force_o[1]

                moment_o[0] = moment_o[0] - yshift * force_o[2]
                moment_o[2] = moment_o[2] + yshift * force_o[0]

                moment_o[0] = moment_o[0] + zshift * force_o[1]
                moment_o[1] = moment_o[1] - zshift * force_o[0]

                force2 = ref2.o2r(force_o)
                moment2 = ref2.o2r(moment_o)

                assert load.loadtype == 'D'
                assert load.name == 'test'
                assert pytest.approx(load.force, 0.0000001) == force2
                assert pytest.approx(load.moment, 0.0000001) == moment2
                assert load.reference == ref2
    
    coord = [-84.2, 0, 3]
    thetas = [-180, -134.5, 0, 45, 105.3]
    shifts = [-23.5, 0, 1.7]
    for x0, y0, z0 in itertools.product(coord, coord, coord):
        for lat, lon, theta in itertools.product(thetas, thetas, thetas): 
            for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
                ref = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                force = np.random.random(3) * 200 - 100
                moment = np.random.random(3) * 200 - 100
                load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)

                direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                                    np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                                    np.sin(np.radians(lat))])
                ref2 = copy.deepcopy(ref)
                ref2.rotate_along(direction, theta)
                ref2.shift(xshift, yshift, zshift)

                load2 = copy.deepcopy(load)
                load.to_reference(ref2)

                load2.rotate_along(direction, theta)
                load2.shift(xshift, yshift, zshift)

                assert load.loadtype == 'D'
                assert load.name == 'test'
                assert pytest.approx(load.force, 0.0000001) == load2.force
                assert pytest.approx(load.moment, 0.0000001) == load2.moment
                assert load.reference == load2.reference


def test_to_origin_shift():
    '''Test to_origin method from Load class for shifts
    '''

    for _ in range(100):
        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load = Load.fromarray('D', name='test', force=force, moment=moment)
        load.to_origin()

        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.0000001) == force
        assert pytest.approx(load.moment, 0.0000001) == moment
        assert load.reference == ReferenceFrame()
    
    # shifts
    coord = [-84.2, -0.4, 0, 3, 18.2]
    shifts = [-23.5, -0.6, 0, 1.7, 1234]
    for x0, y0, z0 in itertools.product(coord, coord, coord):
        for xshift in shifts:
            ref = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            ref.xshift(xshift)
            load.to_origin()

            xshift = -x0
            yshift = -y0
            zshift = -z0
            moment[1] = moment[1] + xshift * force[2]
            moment[2] = moment[2] - xshift * force[1]

            moment[0] = moment[0] - yshift * force[2]
            moment[2] = moment[2] + yshift * force[0]

            moment[0] = moment[0] + zshift * force[1]
            moment[1] = moment[1] - zshift * force[0]

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force
            assert pytest.approx(load.moment, 0.0000001) == moment
            assert load.reference == ReferenceFrame()


def test_to_origin_xrotation():
    '''Test to_origin method from Load class for x rotations
    '''

    # rotation arround x
    thetas = [-180, -134.5, -27, 0, 45, 90, 105.3]
    for theta in thetas:
        ref = ReferenceFrame()
        ref.xrotate(theta)

        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load.to_origin()

        force2 = copy.deepcopy(force)
        moment2 = copy.deepcopy(moment)
        cos = np.cos(np.radians(-theta))
        sin = np.sin(np.radians(-theta))
        force2[1] = cos * force[1] + sin * force[2]
        force2[2] = -sin * force[1] + cos * force[2]
        moment2[1] = cos * moment[1] + sin * moment[2]
        moment2[2] = -sin * moment[1] + cos * moment[2]

        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.0000001) == force2
        assert pytest.approx(load.moment, 0.0000001) == moment2
        assert load.reference == ReferenceFrame()

    for theta in thetas:
        ref = ReferenceFrame()
        ref.xrotate(theta)

        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load2 = copy.deepcopy(load)

        load.to_origin()
        load2.xrotate(-theta)

        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert pytest.approx(load.force, 0.0000001) == load2.force
        assert pytest.approx(load.moment, 0.0000001) == load2.moment
        assert pytest.approx(load.reference.origin, 0.0000001) == load2.reference.origin
        assert pytest.approx(load.reference.xversor, 0.0000001) == load2.reference.xversor
        assert pytest.approx(load.reference.yversor, 0.0000001) == load2.reference.yversor
        assert pytest.approx(load.reference.zversor, 0.0000001) == load2.reference.zversor
    
    for theta in thetas:
        ref = ReferenceFrame()
        ref.xrotate(theta)

        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load.to_origin()

        force2 = ref.r2o(force)
        moment2 = ref.r2o(moment)
        
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.0000001) == force2
        assert pytest.approx(load.moment, 0.0000001) == moment2
        assert load.reference == ReferenceFrame()

    # rotation + displacement
    shifts = [-23.5, -0.6, 0, 1.7, 1234]
    thetas = [-180, -134.5, -27, 0, 45, 90, 105.3]
    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            ref.xrotate(theta)
            ref.shift(xshift, yshift, zshift)

            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load.to_origin()

            force2 = copy.deepcopy(force)
            moment2 = copy.deepcopy(moment)
            cos = np.cos(np.radians(-theta))
            sin = np.sin(np.radians(-theta))
            force2[1] = cos * force[1] + sin * force[2]
            force2[2] = -sin * force[1] + cos * force[2]
            moment2[1] = cos * moment[1] + sin * moment[2]
            moment2[2] = -sin * moment[1] + cos * moment[2]

            xshift = -xshift
            yshift = -yshift
            zshift = -zshift
            moment2[1] = moment2[1] + xshift * force2[2]
            moment2[2] = moment2[2] - xshift * force2[1]

            moment2[0] = moment2[0] - yshift * force2[2]
            moment2[2] = moment2[2] + yshift * force2[0]

            moment2[0] = moment2[0] + zshift * force2[1]
            moment2[1] = moment2[1] - zshift * force2[0]

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force2
            assert pytest.approx(load.moment, 0.0000001) == moment2
            assert load.reference == ReferenceFrame()

    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            ref.xrotate(theta)
            ref.shift(xshift, yshift, zshift)

            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load2 = copy.deepcopy(load)

            load.to_origin()

            load2.xrotate(-theta)
            load2.shift(-xshift, -yshift, -zshift)

            assert load2.loadtype == 'D'
            assert load2.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == load2.force
            assert pytest.approx(load.moment, 0.0000001) == load2.moment
            assert pytest.approx(load.reference.origin, 0.0000001) == load2.reference.origin
            assert pytest.approx(load.reference.xversor, 0.0000001) == load2.reference.xversor
            assert pytest.approx(load.reference.yversor, 0.0000001) == load2.reference.yversor
            assert pytest.approx(load.reference.zversor, 0.0000001) == load2.reference.zversor
    
    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            ref.xrotate(theta)
            ref.shift(xshift, yshift, zshift)
            
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load.to_origin()

            force2 = ref.r2o(force)
            moment2 = ref.r2o(moment)

            xshift = -xshift
            yshift = -yshift
            zshift = -zshift
            moment2[1] = moment2[1] + xshift * force2[2]
            moment2[2] = moment2[2] - xshift * force2[1]

            moment2[0] = moment2[0] - yshift * force2[2]
            moment2[2] = moment2[2] + yshift * force2[0]

            moment2[0] = moment2[0] + zshift * force2[1]
            moment2[1] = moment2[1] - zshift * force2[0]
            
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force2
            assert pytest.approx(load.moment, 0.0000001) == moment2
            assert load.reference == ReferenceFrame()


def test_to_origin_yrotation():
    '''Test to_origin method from Load class for y rotations
    '''

    # rotation arround x
    thetas = [-180, -134.5, -27, 0, 45, 90, 105.3]
    for theta in thetas:
        ref = ReferenceFrame()
        ref.yrotate(theta)

        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load.to_origin()

        force2 = copy.deepcopy(force)
        moment2 = copy.deepcopy(moment)
        cos = np.cos(np.radians(-theta))
        sin = np.sin(np.radians(-theta))
        force2[0] = cos * force[0] - sin * force[2]
        force2[2] = sin * force[0] + cos * force[2]
        moment2[0] = cos * moment[0] - sin * moment[2]
        moment2[2] = sin * moment[0] + cos * moment[2]

        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.0000001) == force2
        assert pytest.approx(load.moment, 0.0000001) == moment2
        assert load.reference == ReferenceFrame()

    for theta in thetas:
        ref = ReferenceFrame()
        ref.yrotate(theta)

        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load2 = copy.deepcopy(load)

        load.to_origin()
        load2.yrotate(-theta)

        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert pytest.approx(load.force, 0.0000001) == load2.force
        assert pytest.approx(load.moment, 0.0000001) == load2.moment
        assert pytest.approx(load.reference.origin, 0.0000001) == load2.reference.origin
        assert pytest.approx(load.reference.xversor, 0.0000001) == load2.reference.xversor
        assert pytest.approx(load.reference.yversor, 0.0000001) == load2.reference.yversor
        assert pytest.approx(load.reference.zversor, 0.0000001) == load2.reference.zversor
    
    for theta in thetas:
        ref = ReferenceFrame()
        ref.yrotate(theta)

        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load.to_origin()

        force2 = ref.r2o(force)
        moment2 = ref.r2o(moment)
        
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.0000001) == force2
        assert pytest.approx(load.moment, 0.0000001) == moment2
        assert load.reference == ReferenceFrame()

    # rotation + displacement
    shifts = [-23.5, -0.6, 0, 1.7, 1234]
    thetas = [-180, -134.5, -27, 0, 45, 90, 105.3]
    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            ref.yrotate(theta)
            ref.shift(xshift, yshift, zshift)

            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load.to_origin()

            force2 = copy.deepcopy(force)
            moment2 = copy.deepcopy(moment)
            cos = np.cos(np.radians(-theta))
            sin = np.sin(np.radians(-theta))
            force2[0] = cos * force[0] - sin * force[2]
            force2[2] = sin * force[0] + cos * force[2]
            moment2[0] = cos * moment[0] - sin * moment[2]
            moment2[2] = sin * moment[0] + cos * moment[2]

            xshift = -xshift
            yshift = -yshift
            zshift = -zshift
            moment2[1] = moment2[1] + xshift * force2[2]
            moment2[2] = moment2[2] - xshift * force2[1]

            moment2[0] = moment2[0] - yshift * force2[2]
            moment2[2] = moment2[2] + yshift * force2[0]

            moment2[0] = moment2[0] + zshift * force2[1]
            moment2[1] = moment2[1] - zshift * force2[0]

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force2
            assert pytest.approx(load.moment, 0.0000001) == moment2
            assert load.reference == ReferenceFrame()

    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            ref.yrotate(theta)
            ref.shift(xshift, yshift, zshift)

            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load2 = copy.deepcopy(load)

            load.to_origin()

            load2.yrotate(-theta)
            load2.shift(-xshift, -yshift, -zshift)

            assert load2.loadtype == 'D'
            assert load2.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == load2.force
            assert pytest.approx(load.moment, 0.0000001) == load2.moment
            assert pytest.approx(load.reference.origin, 0.0000001) == load2.reference.origin
            assert pytest.approx(load.reference.xversor, 0.0000001) == load2.reference.xversor
            assert pytest.approx(load.reference.yversor, 0.0000001) == load2.reference.yversor
            assert pytest.approx(load.reference.zversor, 0.0000001) == load2.reference.zversor
    
    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            ref.yrotate(theta)
            ref.shift(xshift, yshift, zshift)
            
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            
            load.to_origin()

            force2 = ref.r2o(force)
            moment2 = ref.r2o(moment)

            xshift = -xshift
            yshift = -yshift
            zshift = -zshift
            moment2[1] = moment2[1] + xshift * force2[2]
            moment2[2] = moment2[2] - xshift * force2[1]

            moment2[0] = moment2[0] - yshift * force2[2]
            moment2[2] = moment2[2] + yshift * force2[0]

            moment2[0] = moment2[0] + zshift * force2[1]
            moment2[1] = moment2[1] - zshift * force2[0]
            
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force2
            assert pytest.approx(load.moment, 0.0000001) == moment2
            assert load.reference == ReferenceFrame()


def test_to_origin_zrotation():
    '''Test to_origin method from Load class for z rotations
    '''

    # rotation arround x
    thetas = [-180, -134.5, -27, 0, 45, 90, 105.3]
    for theta in thetas:
        ref = ReferenceFrame()
        ref.zrotate(theta)

        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load.to_origin()

        force2 = copy.deepcopy(force)
        moment2 = copy.deepcopy(moment)
        cos = np.cos(np.radians(-theta))
        sin = np.sin(np.radians(-theta))
        force2[0] = cos * force[0] + sin * force[1]
        force2[1] = -sin * force[0] + cos * force[1]
        moment2[0] = cos * moment[0] + sin * moment[1]
        moment2[1] = -sin * moment[0] + cos * moment[1]
        
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.0000001) == force2
        assert pytest.approx(load.moment, 0.0000001) == moment2
        assert load.reference == ReferenceFrame()

    for theta in thetas:
        ref = ReferenceFrame()
        ref.zrotate(theta)

        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load2 = copy.deepcopy(load)

        load.to_origin()
        load2.zrotate(-theta)

        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert pytest.approx(load.force, 0.0000001) == load2.force
        assert pytest.approx(load.moment, 0.0000001) == load2.moment
        assert pytest.approx(load.reference.origin, 0.0000001) == load2.reference.origin
        assert pytest.approx(load.reference.xversor, 0.0000001) == load2.reference.xversor
        assert pytest.approx(load.reference.yversor, 0.0000001) == load2.reference.yversor
        assert pytest.approx(load.reference.zversor, 0.0000001) == load2.reference.zversor
    
    for theta in thetas:
        ref = ReferenceFrame()
        ref.zrotate(theta)

        force = np.random.random(3) * 200 - 100
        moment = np.random.random(3) * 200 - 100
        load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
        load.to_origin()

        force2 = ref.r2o(force)
        moment2 = ref.r2o(moment)
        
        assert load.loadtype == 'D'
        assert load.name == 'test'
        assert pytest.approx(load.force, 0.0000001) == force2
        assert pytest.approx(load.moment, 0.0000001) == moment2
        assert load.reference == ReferenceFrame()

    # rotation + displacement
    shifts = [-23.5, -0.6, 0, 1.7, 1234]
    thetas = [-180, -134.5, -27, 0, 45, 90, 105.3]
    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            ref.zrotate(theta)
            ref.shift(xshift, yshift, zshift)

            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load.to_origin()

            force2 = copy.deepcopy(force)
            moment2 = copy.deepcopy(moment)
            cos = np.cos(np.radians(-theta))
            sin = np.sin(np.radians(-theta))
            force2[0] = cos * force[0] + sin * force[1]
            force2[1] = -sin * force[0] + cos * force[1]
            moment2[0] = cos * moment[0] + sin * moment[1]
            moment2[1] = -sin * moment[0] + cos * moment[1]

            xshift = -xshift
            yshift = -yshift
            zshift = -zshift
            moment2[1] = moment2[1] + xshift * force2[2]
            moment2[2] = moment2[2] - xshift * force2[1]

            moment2[0] = moment2[0] - yshift * force2[2]
            moment2[2] = moment2[2] + yshift * force2[0]

            moment2[0] = moment2[0] + zshift * force2[1]
            moment2[1] = moment2[1] - zshift * force2[0]

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force2
            assert pytest.approx(load.moment, 0.0000001) == moment2
            assert load.reference == ReferenceFrame()

    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            ref.zrotate(theta)
            ref.shift(xshift, yshift, zshift)

            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load2 = copy.deepcopy(load)

            load.to_origin()

            load2.zrotate(-theta)
            load2.shift(-xshift, -yshift, -zshift)

            assert load2.loadtype == 'D'
            assert load2.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == load2.force
            assert pytest.approx(load.moment, 0.0000001) == load2.moment
            assert pytest.approx(load.reference.origin, 0.0000001) == load2.reference.origin
            assert pytest.approx(load.reference.xversor, 0.0000001) == load2.reference.xversor
            assert pytest.approx(load.reference.yversor, 0.0000001) == load2.reference.yversor
            assert pytest.approx(load.reference.zversor, 0.0000001) == load2.reference.zversor
    
    for theta in thetas:
        for xshift, yshift, zshift in itertools.product(shifts, shifts, shifts):
            ref = ReferenceFrame()
            ref.yrotate(theta)
            ref.shift(xshift, yshift, zshift)
            
            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            
            load.to_origin()

            force2 = ref.r2o(force)
            moment2 = ref.r2o(moment)

            xshift = -xshift
            yshift = -yshift
            zshift = -zshift
            moment2[1] = moment2[1] + xshift * force2[2]
            moment2[2] = moment2[2] - xshift * force2[1]

            moment2[0] = moment2[0] - yshift * force2[2]
            moment2[2] = moment2[2] + yshift * force2[0]

            moment2[0] = moment2[0] + zshift * force2[1]
            moment2[1] = moment2[1] - zshift * force2[0]
            
            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force2
            assert pytest.approx(load.moment, 0.0000001) == moment2
            assert load.reference == ReferenceFrame()


def test_to_origin_rotation():
    '''Test to_origin method from Load class for generic
    rotation
    '''

    # general rotation
    coord = [-84.2, -0.4, 0, 3, 18.2]
    thetas = [-180, -134.5, -27, 0, 45, 90, 105.3]
    for x0, y0, z0 in itertools.product(coord, coord, coord):
        for lat, lon, theta in itertools.product(thetas, thetas, thetas):
            direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                                  np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                                  np.sin(np.radians(lat))])

            ref = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            ref.rotate_along(direction, theta)

            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load.to_origin()

            force_o = ref.r2o(force)
            moment_o = ref.r2o(moment)

            xshift = -x0
            yshift = -y0
            zshift = -z0
            moment_o[1] = moment_o[1] + xshift * force_o[2]
            moment_o[2] = moment_o[2] - xshift * force_o[1]

            moment_o[0] = moment_o[0] - yshift * force_o[2]
            moment_o[2] = moment_o[2] + yshift * force_o[0]

            moment_o[0] = moment_o[0] + zshift * force_o[1]
            moment_o[1] = moment_o[1] - zshift * force_o[0]

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force_o
            assert pytest.approx(load.moment, 0.0000001) == moment_o
            assert load.reference == ReferenceFrame()
    
    coord = [-84.2, -0.4, 0, 3, 18.2]
    thetas = [-180, -134.5, -27, 0, 45, 90, 105.3]
    for x0, y0, z0 in itertools.product(coord, coord, coord):
        for lat, lon, theta in itertools.product(thetas, thetas, thetas): 
            direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                                  np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                                  np.sin(np.radians(lat))])

            ref = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            ref.rotate_along(direction, theta)

            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load2 = copy.deepcopy(load)
            load.to_origin()

            
            load2.rotate_along(direction, -theta)
            load2.shift(-x0, -y0, -z0)

            assert load2.loadtype == 'D'
            assert load2.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == load2.force
            assert pytest.approx(load.moment, 0.0000001) == load2.moment
            assert pytest.approx(load.reference.origin, 0.0000001) == load2.reference.origin
            assert pytest.approx(load.reference.xversor, 0.0000001) == load2.reference.xversor
            assert pytest.approx(load.reference.yversor, 0.0000001) == load2.reference.yversor
            assert pytest.approx(load.reference.zversor, 0.0000001) == load2.reference.zversor


def test_resetorigin():
    '''Test resetorigin method from Load class
    '''
    ref0 = ReferenceFrame()
    coord = [-84.2, -0.4, 0, 3, 18.2]
    thetas = [-180, -134.5, -27, 0, 45, 90, 105.3]
    for x0, y0, z0 in itertools.product(coord, coord, coord):
        for lat, lon, theta in itertools.product(thetas, thetas, thetas): 
            direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                                  np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                                  np.sin(np.radians(lat))])

            ref = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            ref.rotate_along(direction, theta)

            force = np.random.random(3) * 200 - 100
            moment = np.random.random(3) * 200 - 100
            load = Load.fromarray('D', name='test', force=force, moment=moment, reference=ref)
            load.resetorigin()

            assert load.loadtype == 'D'
            assert load.name == 'test'
            assert pytest.approx(load.force, 0.0000001) == force
            assert pytest.approx(load.moment, 0.0000001) == moment
            assert pytest.approx(load.reference.origin, 0.0000001) == ref0.origin
            assert pytest.approx(load.reference.xversor, 0.0000001) == ref0.xversor
            assert pytest.approx(load.reference.yversor, 0.0000001) == ref0.yversor
            assert pytest.approx(load.reference.zversor, 0.0000001) == ref0.zversor


def test_add_nan():
    '''Test _add__ method from Load class for nans inputs
    '''

    # nan handling
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        force1 = np.random.random(3) * 200 - 100
        moment1 = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('D', force=force1, moment=moment1, reference=ref)

        force2 = [np.nan, np.nan, np.nan]
        moment2 = [np.nan, np.nan, np.nan]
        load2 = Load.fromarray('D', force=force2, moment=moment2, reference=ref)
        load3 = load1 + load2
        assert all(np.isnan(load3.force))
        assert all(np.isnan(load3.moment))
        assert load3.reference == ref


        force2 = np.random.random(3)
        moment2 = [np.nan, np.nan, np.nan]
        load2 = Load.fromarray('D', force=force2, moment=moment2, reference=ref)
        load3 = load1 + load2
        assert all(np.isnan(load3.force))
        assert all(np.isnan(load3.moment))
        assert load3.reference == ref


        force2 = [np.nan, np.nan, np.nan]
        moment2 = np.random.random(3)
        load2 = Load.fromarray('D', force=force2, moment=moment2, reference=ref)
        load3 = load1 + load2
        assert all(np.isnan(load3.force))
        assert all(np.isnan(load3.moment))
        assert load3.reference == ref


        force2 = [np.nan, 3, 4]
        moment2 = np.random.random(3)
        load2 = Load.fromarray('D', force=force2, moment=moment2, reference=ref)
        load3 = load1 + load2
        assert all(np.isnan(load3.force))
        assert all(np.isnan(load3.moment))
        assert load3.reference == ref

        force2 = [np.nan, 3, np.nan]
        moment2 = np.random.random(3)
        load2 = Load.fromarray('D', force=force2, moment=moment2, reference=ref)
        load3 = load1 + load2
        assert all(np.isnan(load3.force))
        assert all(np.isnan(load3.moment))
        assert load3.reference == ref

        force2 = np.random.random(3)
        moment2 = [np.nan, 3, 4]
        load2 = Load.fromarray('D', force=force2, moment=moment2, reference=ref)
        load3 = load1 + load2
        assert all(np.isnan(load3.force))
        assert all(np.isnan(load3.moment))
        assert load3.reference == ref

        force2 = np.random.random(3)
        moment2 = [np.nan, np.nan, 4]
        load2 = Load.fromarray('D', force=force2, moment=moment2, reference=ref)
        load3 = load1 + load2
        assert all(np.isnan(load3.force))
        assert all(np.isnan(load3.moment))
        assert load3.reference == ref

    # wrong input type
    with pytest.raises(TypeError):
        force1 = np.random.random(3) * 200 - 100
        moment1 = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('D', force=force1, moment=moment1, reference=ref)
        load2 = 3
        load3 = load1 + load2

    with pytest.raises(TypeError):
        force1 = np.random.random(3) * 200 - 100
        moment1 = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('D', force=force1, moment=moment1, reference=ref)
        load2 = [1, 2, 3]
        load3 = load1 + load2
    
    with pytest.raises(TypeError):
        force1 = np.random.random(3) * 200 - 100
        moment1 = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('D', force=force1, moment=moment1, reference=ref)
        load2 = 'load'
        load3 = load1 + load2


def test_add_different_datatype():
    '''Test __add__ method from Load class for different data
    types
    '''

    force = [[1, 2, 3], np.array([1, 2, 3]), [1.23, 2.42, 3.12], np.array([1.23, 2.42, 3.12])]
    moment = [[1, 2, 3], np.array([1, 2, 3]), [1.23, 2.42, 3.12], np.array([1.23, 2.42, 3.12])]
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    refs = [ref1, ref2, ref3]

    for force1, force2, moment1, moment2 in itertools.product(force, force, moment, moment):
        for ref in refs:
    
            load1 = Load.fromarray('D', force=force1, moment=moment1, reference=ref)
            load2 = Load.fromarray('D', force=force2, moment=moment2, reference=ref)

            load3 = load1 + load2

            force1 = np.array(force1)
            force2 = np.array(force2)

            moment1 = np.array(moment1)
            moment2 = np.array(moment2)

            # no changes in original loads
            assert load1.loadtype == 'D'
            assert pytest.approx(load1.force, 0.0000001) == force1
            assert pytest.approx(load1.moment, 0.0000001) == moment1
            assert load1.reference == ref

            assert load2.loadtype == 'D'
            assert pytest.approx(load2.force, 0.0000001) == force2
            assert pytest.approx(load2.moment, 0.0000001) == moment2
            assert load2.reference == ref

            # new load is the sum
            assert load3.loadtype == 'D'
            assert pytest.approx(load3.force, 0.0000001) == force1 + force2
            assert pytest.approx(load3.moment, 0.0000001) == moment1 + moment2
            assert load3.reference == ref
    

    vals = [0, 1, -1.2323]
    for xforce1, yforce1, zforce1, xmoment1, ymoment1, zmoment1 in itertools.product(vals, vals, vals, vals, vals, vals):
        for xforce2, yforce2, zforce2, xmoment2, ymoment2, zmoment2 in itertools.product(vals, vals, vals, vals, vals, vals):

            load1 = Load('D', xforce=xforce1, yforce=yforce1, zforce=zforce1,
                                xmoment=xmoment1, ymoment=ymoment1, zmoment=zmoment1, reference=ref)
            load2 = Load('D', xforce=xforce2, yforce=yforce2, zforce=zforce2,
                                xmoment=xmoment2, ymoment=ymoment2, zmoment=zmoment2, reference=ref)

            load3 = load1 + load2

            force1 = np.array([xforce1, yforce1, zforce1])
            force2 = np.array([xforce2, yforce2, zforce2])

            moment1 = np.array([xmoment1, ymoment1, zmoment1])
            moment2 = np.array([xmoment2, ymoment2, zmoment2])

            # no changes in original loads
            assert load1.loadtype == 'D'
            assert pytest.approx(load1.force, 0.0000001) == force1
            assert pytest.approx(load1.moment, 0.0000001) == moment1
            assert load1.reference == ref

            assert load2.loadtype == 'D'
            assert pytest.approx(load2.force, 0.0000001) == force2
            assert pytest.approx(load2.moment, 0.0000001) == moment2
            assert load2.reference == ref

            # new load is the sum
            assert load3.loadtype == 'D'
            assert pytest.approx(load3.force, 0.0000001) == force1 + force2
            assert pytest.approx(load3.moment, 0.0000001) == moment1 + moment2
            assert load3.reference == ref


def test_add_loadtype_behavior():
    '''Test __add__ method from Load class for different data
    types
    '''

    force1 = np.random.random(3)
    force2 = np.random.random(3)
    moment1 = np.random.random(3)
    moment2 = np.random.random(3)

    loadtype1 = 'D'
    for loadtype2 in LOAD:
        load1 = Load.fromarray(loadtype1, name='xasdq', force=force1, moment=moment1)
        load2 = Load.fromarray(loadtype2, name='iwqe', force=force2, moment=moment2)

        load3 = load1 + load2

        assert load1.loadtype == loadtype1
        assert load1.name == 'xasdq'
        assert pytest.approx(load1.force, 0.0000001) == force1
        assert pytest.approx(load1.moment, 0.0000001) == moment1
        assert load3.reference == ReferenceFrame()

        assert load2.loadtype == loadtype2
        assert load2.name == 'iwqe'
        assert pytest.approx(load2.force, 0.0000001) == force2
        assert pytest.approx(load2.moment, 0.0000001) == moment2
        assert load2.reference == ReferenceFrame()

        assert load3.loadtype == loadtype1
        assert load3.name == 'xasdq'
        assert pytest.approx(load3.force, 0.0000001) == force1 + force2
        assert pytest.approx(load3.moment, 0.0000001) == moment1 + moment2
        assert load2.reference == ReferenceFrame()

        # remains unchaned when changing the input loads
        load1.shift(1, 2, 3)
        load1.rotate_along([1, 2, 4], 34)

        load2.shift(-2, 3, -1)
        load2.xrotate(23)

        assert load3.loadtype == loadtype1
        assert load3.name == 'xasdq'
        assert pytest.approx(load3.force, 0.0000001) == force1 + force2
        assert pytest.approx(load3.moment, 0.0000001) == moment1 + moment2


def test_add_same_ref():
    '''Test _add__ method from Load class for same reference
    system
    '''

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for _ in range(100):
            force1 = np.random.random(3) * 200 - 100
            moment1 = np.random.random(3) * 200 - 100
            load1 = Load.fromarray('D', name='test', force=force1, moment=moment1, reference=ref)

            force2 = np.random.random(3) * 200 - 100
            moment2 = np.random.random(3) * 200 - 100
            load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref)

            load3 = load1 + load2


            # no changes in original loads
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.0000001) == force1
            assert pytest.approx(load1.moment, 0.0000001) == moment1
            assert load1.reference == ref

            assert load2.loadtype == 'D'
            assert load2.name == 'test'
            assert pytest.approx(load2.force, 0.0000001) == force2
            assert pytest.approx(load2.moment, 0.0000001) == moment2
            assert load2.reference == ref

            # new load is the sum
            assert load3.loadtype == 'D'
            assert load3.name == 'test'
            assert pytest.approx(load3.force, 0.0000001) == force1 + force2
            assert pytest.approx(load3.moment, 0.0000001) == moment1 + moment2
            assert load3.reference == ref

            # changes in the original laods do not affect the sum afterwards
            load1.shift(3, 1, 2)
            load1.xrotate(14)
            load1.yrotate(-34)

            load2.shift(-4, 4, 4.2)

            assert load3.loadtype == 'D'
            assert load3.name == 'test'
            assert pytest.approx(load3.force, 0.0000001) == force1 + force2
            assert pytest.approx(load3.moment, 0.0000001) == moment1 + moment2
            assert load3.reference == ref


def test_add_different_ref():
    '''Test _add__ method from Load class for different reference
    systems
    '''

    coord = [-84.2, -0.4, 0, 3, 18.2]
    thetas = [-180, -134.5, -27, 0, 45, 90, 105.3]

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for x0, y0, z0 in itertools.product(coord, coord, coord):
            for lat, lon, theta in itertools.product(thetas, thetas, thetas): 
                direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                                    np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                                    np.sin(np.radians(lat))])

                ref_load2 = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                ref_load2.rotate_along(direction, theta)

                force1 = np.random.random(3) * 200 - 100
                moment1 = np.random.random(3) * 200 - 100
                load1 = Load.fromarray('D', name='test', force=force1, moment=moment1, reference=ref)

                force2 = np.random.random(3) * 200 - 100
                moment2 = np.random.random(3) * 200 - 100
                load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref_load2)

                load3 = load1 + load2

                load2_ref = copy.deepcopy(load2)
                load2_ref.to_reference(ref)


                # no changes in original loads
                assert load1.loadtype == 'D'
                assert load1.name == 'test'
                assert pytest.approx(load1.force, 0.0000001) == force1
                assert pytest.approx(load1.moment, 0.0000001) == moment1
                assert load1.reference == ref

                assert load2.loadtype == 'D'
                assert load2.name == 'test'
                assert pytest.approx(load2.force, 0.0000001) == force2
                assert pytest.approx(load2.moment, 0.0000001) == moment2
                assert load2.reference == ref_load2

                # new load is the sum
                assert load3.loadtype == 'D'
                assert load3.name == 'test'
                assert pytest.approx(load3.force, 0.0000001) == force1 + load2_ref.force
                assert pytest.approx(load3.moment, 0.0000001) == moment1 + load2_ref.moment
                assert load3.reference == ref

                # changes in the original laods do not affect the sum afterwards
                load1.shift(3, 1, 2)
                load1.xrotate(14)
                load1.yrotate(-34)

                load2.shift(-4, 4, 4.2)

                assert load3.loadtype == 'D'
                assert load3.name == 'test'
                assert pytest.approx(load3.force, 0.0000001) == force1 + load2_ref.force
                assert pytest.approx(load3.moment, 0.0000001) == moment1 + load2_ref.moment
                assert load3.reference == ref


def test_sub_nan():
    '''Test __sub__ method from Load class for nans inputs
    '''

    # nan handling
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        force1 = np.random.random(3) * 200 - 100
        moment1 = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('D', force=force1, moment=moment1, reference=ref)

        force2 = [np.nan, np.nan, np.nan]
        moment2 = [np.nan, np.nan, np.nan]
        load2 = Load.fromarray('D', force=force2, moment=moment2, reference=ref)
        load3 = load1 - load2
        assert all(np.isnan(load3.force))
        assert all(np.isnan(load3.moment))
        assert load3.reference == ref


        force2 = np.random.random(3)
        moment2 = [np.nan, np.nan, np.nan]
        load2 = Load.fromarray('D', force=force2, moment=moment2, reference=ref)
        load3 = load1 - load2
        assert all(np.isnan(load3.force))
        assert all(np.isnan(load3.moment))
        assert load3.reference == ref


        force2 = [np.nan, np.nan, np.nan]
        moment2 = np.random.random(3)
        load2 = Load.fromarray('D', force=force2, moment=moment2, reference=ref)
        load3 = load1 - load2
        assert all(np.isnan(load3.force))
        assert all(np.isnan(load3.moment))
        assert load3.reference == ref


        force2 = [np.nan, 3, 4]
        moment2 = np.random.random(3)
        load2 = Load.fromarray('D', force=force2, moment=moment2, reference=ref)
        load3 = load1 - load2
        assert all(np.isnan(load3.force))
        assert all(np.isnan(load3.moment))
        assert load3.reference == ref

        force2 = [np.nan, 3, np.nan]
        moment2 = np.random.random(3)
        load2 = Load.fromarray('D', force=force2, moment=moment2, reference=ref)
        load3 = load1 - load2
        assert all(np.isnan(load3.force))
        assert all(np.isnan(load3.moment))
        assert load3.reference == ref

        force2 = np.random.random(3)
        moment2 = [np.nan, 3, 4]
        load2 = Load.fromarray('D', force=force2, moment=moment2, reference=ref)
        load3 = load1 - load2
        assert all(np.isnan(load3.force))
        assert all(np.isnan(load3.moment))
        assert load3.reference == ref

        force2 = np.random.random(3)
        moment2 = [np.nan, np.nan, 4]
        load2 = Load.fromarray('D', force=force2, moment=moment2, reference=ref)
        load3 = load1 - load2
        assert all(np.isnan(load3.force))
        assert all(np.isnan(load3.moment))
        assert load3.reference == ref

    # wrong input type
    with pytest.raises(TypeError):
        force1 = np.random.random(3) * 200 - 100
        moment1 = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('D', force=force1, moment=moment1, reference=ref)
        load2 = 3
        load3 = load1 - load2

    with pytest.raises(TypeError):
        force1 = np.random.random(3) * 200 - 100
        moment1 = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('D', force=force1, moment=moment1, reference=ref)
        load2 = [1, 2, 3]
        load3 = load1 - load2
    
    with pytest.raises(TypeError):
        force1 = np.random.random(3) * 200 - 100
        moment1 = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('D', force=force1, moment=moment1, reference=ref)
        load2 = 'load'
        load3 = load1 - load2


def test_sub_different_datatype():
    '''Test __sub__ method from Load class for different data
    types
    '''

    force = [[1, 2, 3], np.array([1, 2, 3]), [1.23, 2.42, 3.12], np.array([1.23, 2.42, 3.12])]
    moment = [[1, 2, 3], np.array([1, 2, 3]), [1.23, 2.42, 3.12], np.array([1.23, 2.42, 3.12])]
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    refs = [ref1, ref2, ref3]

    for force1, force2, moment1, moment2 in itertools.product(force, force, moment, moment):
        for ref in refs:
    
            load1 = Load.fromarray('D', force=force1, moment=moment1, reference=ref)
            load2 = Load.fromarray('D', force=force2, moment=moment2, reference=ref)

            load3 = load1 - load2

            force1 = np.array(force1)
            force2 = np.array(force2)

            moment1 = np.array(moment1)
            moment2 = np.array(moment2)

            # no changes in original loads
            assert load1.loadtype == 'D'
            assert pytest.approx(load1.force, 0.0000001) == force1
            assert pytest.approx(load1.moment, 0.0000001) == moment1
            assert load1.reference == ref

            assert load2.loadtype == 'D'
            assert pytest.approx(load2.force, 0.0000001) == force2
            assert pytest.approx(load2.moment, 0.0000001) == moment2
            assert load2.reference == ref

            # new load is the sum
            assert load3.loadtype == 'D'
            assert pytest.approx(load3.force, 0.0000001) == force1 - force2
            assert pytest.approx(load3.moment, 0.0000001) == moment1 - moment2
            assert load3.reference == ref
    

    vals = [0, 1, -1.2323]
    for xforce1, yforce1, zforce1, xmoment1, ymoment1, zmoment1 in itertools.product(vals, vals, vals, vals, vals, vals):
        for xforce2, yforce2, zforce2, xmoment2, ymoment2, zmoment2 in itertools.product(vals, vals, vals, vals, vals, vals):

            load1 = Load('D', xforce=xforce1, yforce=yforce1, zforce=zforce1,
                                xmoment=xmoment1, ymoment=ymoment1, zmoment=zmoment1, reference=ref)
            load2 = Load('D', xforce=xforce2, yforce=yforce2, zforce=zforce2,
                                xmoment=xmoment2, ymoment=ymoment2, zmoment=zmoment2, reference=ref)

            load3 = load1 - load2

            force1 = np.array([xforce1, yforce1, zforce1])
            force2 = np.array([xforce2, yforce2, zforce2])

            moment1 = np.array([xmoment1, ymoment1, zmoment1])
            moment2 = np.array([xmoment2, ymoment2, zmoment2])

            # no changes in original loads
            assert load1.loadtype == 'D'
            assert pytest.approx(load1.force, 0.0000001) == force1
            assert pytest.approx(load1.moment, 0.0000001) == moment1
            assert load1.reference == ref

            assert load2.loadtype == 'D'
            assert pytest.approx(load2.force, 0.0000001) == force2
            assert pytest.approx(load2.moment, 0.0000001) == moment2
            assert load2.reference == ref

            # new load is the sum
            assert load3.loadtype == 'D'
            assert pytest.approx(load3.force, 0.0000001) == force1 - force2
            assert pytest.approx(load3.moment, 0.0000001) == moment1 - moment2
            assert load3.reference == ref


def test_sub_loadtype_behavior():
    '''Test __sub__ method from Load class for different data
    types
    '''

    force1 = np.random.random(3)
    force2 = np.random.random(3)
    moment1 = np.random.random(3)
    moment2 = np.random.random(3)

    loadtype1 = 'D'
    for loadtype2 in LOAD:
        load1 = Load.fromarray(loadtype1, name='xasdq', force=force1, moment=moment1)
        load2 = Load.fromarray(loadtype2, name='iwqe', force=force2, moment=moment2)

        load3 = load1 - load2

        assert load1.loadtype == loadtype1
        assert load1.name == 'xasdq'
        assert pytest.approx(load1.force, 0.0000001) == force1
        assert pytest.approx(load1.moment, 0.0000001) == moment1
        assert load3.reference == ReferenceFrame()

        assert load2.loadtype == loadtype2
        assert load2.name == 'iwqe'
        assert pytest.approx(load2.force, 0.0000001) == force2
        assert pytest.approx(load2.moment, 0.0000001) == moment2
        assert load2.reference == ReferenceFrame()

        assert load3.loadtype == loadtype1
        assert load3.name == 'xasdq'
        assert pytest.approx(load3.force, 0.0000001) == force1 - force2
        assert pytest.approx(load3.moment, 0.0000001) == moment1 - moment2
        assert load2.reference == ReferenceFrame()

        # remains unchaned when changing the input loads
        load1.shift(1, 2, 3)
        load1.rotate_along([1, 2, 4], 34)

        load2.shift(-2, 3, -1)
        load2.xrotate(23)

        assert load3.loadtype == loadtype1
        assert load3.name == 'xasdq'
        assert pytest.approx(load3.force, 0.0000001) == force1 - force2
        assert pytest.approx(load3.moment, 0.0000001) == moment1 - moment2


def test_sub_same_ref():
    '''Test __sub__ method from Load class for same reference
    system
    '''

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for _ in range(100):
            force1 = np.random.random(3) * 200 - 100
            moment1 = np.random.random(3) * 200 - 100
            load1 = Load.fromarray('D', name='test', force=force1, moment=moment1, reference=ref)

            force2 = np.random.random(3) * 200 - 100
            moment2 = np.random.random(3) * 200 - 100
            load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref)

            load3 = load1 - load2


            # no changes in original loads
            assert load1.loadtype == 'D'
            assert load1.name == 'test'
            assert pytest.approx(load1.force, 0.0000001) == force1
            assert pytest.approx(load1.moment, 0.0000001) == moment1
            assert load1.reference == ref

            assert load2.loadtype == 'D'
            assert load2.name == 'test'
            assert pytest.approx(load2.force, 0.0000001) == force2
            assert pytest.approx(load2.moment, 0.0000001) == moment2
            assert load2.reference == ref

            # new load is the sum
            assert load3.loadtype == 'D'
            assert load3.name == 'test'
            assert pytest.approx(load3.force, 0.0000001) == force1 - force2
            assert pytest.approx(load3.moment, 0.0000001) == moment1 - moment2
            assert load3.reference == ref

            # changes in the original laods do not affect the sum afterwards
            load1.shift(3, 1, 2)
            load1.xrotate(14)
            load1.yrotate(-34)

            load2.shift(-4, 4, 4.2)

            assert load3.loadtype == 'D'
            assert load3.name == 'test'
            assert pytest.approx(load3.force, 0.0000001) == force1 - force2
            assert pytest.approx(load3.moment, 0.0000001) == moment1 - moment2
            assert load3.reference == ref


def test_sub_different_ref():
    '''Test _add__ method from Load class for different reference
    systems
    '''

    coord = [-84.2, -0.4, 0, 3, 18.2]
    thetas = [-180, -134.5, -27, 0, 45, 90, 105.3]

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for x0, y0, z0 in itertools.product(coord, coord, coord):
            for lat, lon, theta in itertools.product(thetas, thetas, thetas): 
                direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                                    np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                                    np.sin(np.radians(lat))])

                ref_load2 = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                ref_load2.rotate_along(direction, theta)

                force1 = np.random.random(3) * 200 - 100
                moment1 = np.random.random(3) * 200 - 100
                load1 = Load.fromarray('D', name='test', force=force1, moment=moment1, reference=ref)

                force2 = np.random.random(3) * 200 - 100
                moment2 = np.random.random(3) * 200 - 100
                load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref_load2)

                load3 = load1 - load2

                load2_ref = copy.deepcopy(load2)
                load2_ref.to_reference(ref)


                # no changes in original loads
                assert load1.loadtype == 'D'
                assert load1.name == 'test'
                assert pytest.approx(load1.force, 0.0000001) == force1
                assert pytest.approx(load1.moment, 0.0000001) == moment1
                assert load1.reference == ref

                assert load2.loadtype == 'D'
                assert load2.name == 'test'
                assert pytest.approx(load2.force, 0.0000001) == force2
                assert pytest.approx(load2.moment, 0.0000001) == moment2
                assert load2.reference == ref_load2

                # new load is the sum
                assert load3.loadtype == 'D'
                assert load3.name == 'test'
                assert pytest.approx(load3.force, 0.0000001) == force1 - load2_ref.force
                assert pytest.approx(load3.moment, 0.0000001) == moment1 - load2_ref.moment
                assert load3.reference == ref

                # changes in the original laods do not affect the sum afterwards
                load1.shift(3, 1, 2)
                load1.xrotate(14)
                load1.yrotate(-34)

                load2.shift(-4, 4, 4.2)

                assert load3.loadtype == 'D'
                assert load3.name == 'test'
                assert pytest.approx(load3.force, 0.0000001) == force1 - load2_ref.force
                assert pytest.approx(load3.moment, 0.0000001) == moment1 - load2_ref.moment
                assert load3.reference == ref


def test_mul():
    '''Test __mul__ method from Load class'''

    # test nan
    force = [1, 2, 3]
    moment = [3, 4, 5]
    load = Load.fromarray('L', name='cucu', force=force, moment=moment)
    load2 = np.nan * load
    assert all(np.isnan(load2.force))
    assert all(np.isnan(load2.moment))
    assert load2.reference == ReferenceFrame()

    load2 = load * np.nan
    assert all(np.isnan(load2.force))
    assert all(np.isnan(load2.moment))
    assert load2.reference == ReferenceFrame()

    # test wrong data type
    with pytest.raises(TypeError):
        load2 = '3' * load
    
    with pytest.raises(TypeError):
        load2 = load * '3'

    with pytest.raises(TypeError):
        load2 = load * [3]

    with pytest.raises(TypeError):
        load2 = [3] * load

    with pytest.raises(TypeError):
        load2 = load * np.array([3])
    
    with pytest.raises(TypeError):
        load2 = np.array([3]) * load

    # original load not modified after product
    force = np.array([1, 2, 3])
    moment = np.array([3, 4, 5])
    load = Load.fromarray('L', name='cucu', force=force, moment=moment)
    load2 = 3 * load
    assert load.loadtype == 'L'
    assert load.name == 'cucu'
    assert pytest.approx(load.force, 0.0000001) == force
    assert pytest.approx(load.moment, 0.0000001) == moment
    assert load.reference == ReferenceFrame()

    assert load.loadtype == 'L'
    assert load.name == 'cucu'

    # original load not modified by alterations ot sencondary load
    load2.shift(3, 1, 2)
    load2.xrotate(34)
    assert load.loadtype == 'L'
    assert load.name == 'cucu'
    assert pytest.approx(load.force, 0.0000001) == force
    assert pytest.approx(load.moment, 0.0000001) == moment
    assert load.reference == ReferenceFrame()

    vals = [-123, 0, 0.5, 1, 101235]
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for xforce, yforce, zforce, xmoment, ymoment, zmoment in itertools.product(vals, vals, vals, vals, vals, vals):
            for multby in vals:
                load1 = Load('L', name='xyz', xforce=xforce, yforce=yforce, zforce=zforce,
                                    xmoment=xmoment, ymoment=ymoment, zmoment=zmoment,
                                    reference=ref)
                load2 = load1 * multby

                force = np.array([xforce, yforce, zforce])
                moment = np.array([xmoment, ymoment, zmoment])

                assert load1.loadtype == 'L'
                assert load1.name == 'xyz'
                assert pytest.approx(load1.force, 0.0000001) == force
                assert pytest.approx(load1.moment, 0.0000001) == moment
                assert load1.reference == ref

                assert load2.loadtype == 'L'
                assert load2.name == 'xyz'
                assert pytest.approx(load2.force, 0.0000001) == force * multby
                assert pytest.approx(load2.moment, 0.0000001) == moment * multby
                assert load2.reference == ref

                load2 = multby * load1

                assert load1.loadtype == 'L'
                assert load1.name == 'xyz'
                assert pytest.approx(load1.force, 0.0000001) == force
                assert pytest.approx(load1.moment, 0.0000001) == moment
                assert load1.reference == ref

                assert load2.loadtype == 'L'
                assert load2.name == 'xyz'
                assert pytest.approx(load2.force, 0.0000001) == force * multby
                assert pytest.approx(load2.moment, 0.0000001) == moment * multby
                assert load2.reference == ref


def test_truediv():
    '''Test __truediv__ method from Load class'''


    # test nan
    force = [1, 2, 3]
    moment = [3, 4, 5]
    load = Load.fromarray('W', name='yui', force=force, moment=moment)

    load2 = load / np.nan
    assert all(np.isnan(load2.force))
    assert all(np.isnan(load2.moment))
    assert load2.reference == ReferenceFrame()

    # test wrong data type
    with pytest.raises(TypeError):
        load2 = load / '3'

    with pytest.raises(TypeError):
        load2 = load / [3]
    
    with pytest.raises(TypeError):
        load2 = load / np.array([3])

    # test division by zero
    with pytest.raises(ValueError):
        load2 = load / 0

    vals = [-123, 0, 0.5, 1, 101235]
    divbys = [-123, 0.5, 1, 101235]
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for xforce, yforce, zforce, xmoment, ymoment, zmoment in itertools.product(vals, vals, vals, vals, vals, vals):
            for divby in divbys:
                load1 = Load('W', name='yui', xforce=xforce, yforce=yforce, zforce=zforce,
                                    xmoment=xmoment, ymoment=ymoment, zmoment=zmoment,
                                    reference=ref)
                load2 = load1 / divby

                force = np.array([xforce, yforce, zforce])
                moment = np.array([xmoment, ymoment, zmoment])

                assert load1.loadtype == 'W'
                assert load1.name == 'yui'
                assert pytest.approx(load1.force, 0.0000001) == force
                assert pytest.approx(load1.moment, 0.0000001) == moment
                assert load1.reference == ref

                assert load2.loadtype == 'W'
                assert load2.name == 'yui'
                assert pytest.approx(load2.force, 0.0000001) == force / divby
                assert pytest.approx(load2.moment, 0.0000001) == moment / divby
                assert load2.reference == ref


def test_iadd_inputs():
    '''Test __iadd__ method from Load class for different inputs
    '''

    # nan handling
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        force1 = np.random.random(3) * 200 - 100
        moment1 = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('W', name='id', force=force1, moment=moment1, reference=ref)

        force2 = [np.nan, np.nan, np.nan]
        moment2 = [np.nan, np.nan, np.nan]
        load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref)
        load1 += load2
        assert load1.loadtype == 'W'
        assert load1.name == 'id'
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert all(np.isnan(load1.force))
        assert all(np.isnan(load1.moment))
        assert load1.reference == ref


        force2 = np.random.random(3)
        moment2 = [np.nan, np.nan, np.nan]
        load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref)
        load1 += load2
        assert load1.loadtype == 'W'
        assert load1.name == 'id'
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert all(np.isnan(load1.force))
        assert all(np.isnan(load1.moment))
        assert load1.reference == ref


        force2 = [np.nan, np.nan, np.nan]
        moment2 = np.random.random(3)
        load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref)
        load1 += load2
        assert load1.loadtype == 'W'
        assert load1.name == 'id'
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert all(np.isnan(load1.force))
        assert all(np.isnan(load1.moment))
        assert load1.reference == ref


        force2 = [np.nan, 3, 4]
        moment2 = np.random.random(3)
        load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref)
        load1 += load2
        assert load1.loadtype == 'W'
        assert load1.name == 'id'
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert all(np.isnan(load1.force))
        assert all(np.isnan(load1.moment))
        assert load1.reference == ref

        force2 = [np.nan, 3, np.nan]
        moment2 = np.random.random(3)
        load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref)
        load1 += load2
        assert load1.loadtype == 'W'
        assert load1.name == 'id'
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert all(np.isnan(load1.force))
        assert all(np.isnan(load1.moment))
        assert load1.reference == ref

        force2 = np.random.random(3)
        moment2 = [np.nan, 3, 4]
        load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref)
        load1 += load2
        assert load1.loadtype == 'W'
        assert load1.name == 'id'
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert all(np.isnan(load1.force))
        assert all(np.isnan(load1.moment))
        assert load1.reference == ref

        force2 = np.random.random(3)
        moment2 = [np.nan, np.nan, 4]
        load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref)
        load1 += load2
        assert load1.loadtype == 'W'
        assert load1.name == 'id'
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert all(np.isnan(load1.force))
        assert all(np.isnan(load1.moment))
        assert load1.reference == ref

    # wrong input type
    with pytest.raises(TypeError):
        force1 = np.random.random(3) * 200 - 100
        moment1 = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('D', name='test', force=force1, moment=moment1, reference=ref)
        load1 += 1
    
    with pytest.raises(TypeError):
        force1 = np.random.random(3) * 200 - 100
        moment1 = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('D', name='test', force=force1, moment=moment1, reference=ref)
        load1 += [1, 2, 3]
    
    with pytest.raises(TypeError):
        force1 = np.random.random(3) * 200 - 100
        moment1 = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('D', name='test', force=force1, moment=moment1, reference=ref)
        load1 += 'load'

    # different data types
    force = [[1, 2, 3], np.array([1, 2, 3]), [1.23, 2.42, 3.12], np.array([1.23, 2.42, 3.12])]
    moment = [[1, 2, 3], np.array([1, 2, 3]), [1.23, 2.42, 3.12], np.array([1.23, 2.42, 3.12])]
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    refs = [ref1, ref2, ref3]

    for force1, force2, moment1, moment2 in itertools.product(force, force, moment, moment):
        for ref in refs:
    
            load1 = Load.fromarray('W', name='id', force=force1, moment=moment1, reference=ref)
            load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref)

            load1 += load2

            force1 = np.array(force1)
            force2 = np.array(force2)

            moment1 = np.array(moment1)
            moment2 = np.array(moment2)

            assert load2.loadtype == 'D'
            assert load2.name == 'test'
            assert pytest.approx(load2.force, 0.0000001) == force2
            assert pytest.approx(load2.moment, 0.0000001) == moment2
            assert load2.reference == ref

            # new load is the sum
            assert load1.loadtype == 'W'
            assert load1.name == 'id'
            assert pytest.approx(load1.force, 0.0000001) == force1 + force2
            assert pytest.approx(load1.moment, 0.0000001) == moment1 + moment2
            assert load1.reference == ref
    

    vals = [0, 1, -1.2323]
    for xforce1, yforce1, zforce1, xmoment1, ymoment1, zmoment1 in itertools.product(vals, vals, vals, vals, vals, vals):
        for xforce2, yforce2, zforce2, xmoment2, ymoment2, zmoment2 in itertools.product(vals, vals, vals, vals, vals, vals):

            load1 = Load('W', name='id', xforce=xforce1, yforce=yforce1, zforce=zforce1,
                                xmoment=xmoment1, ymoment=ymoment1, zmoment=zmoment1, reference=ref)
            load2 = Load('D', name='test', xforce=xforce2, yforce=yforce2, zforce=zforce2,
                                xmoment=xmoment2, ymoment=ymoment2, zmoment=zmoment2, reference=ref)

            load1 += load2

            force1 = np.array([xforce1, yforce1, zforce1])
            force2 = np.array([xforce2, yforce2, zforce2])

            moment1 = np.array([xmoment1, ymoment1, zmoment1])
            moment2 = np.array([xmoment2, ymoment2, zmoment2])

            # no changes in original loads
            assert load2.loadtype == 'D'
            assert load2.name == 'test'
            assert pytest.approx(load2.force, 0.0000001) == force2
            assert pytest.approx(load2.moment, 0.0000001) == moment2
            assert load2.reference == ref

            # new load is the sum
            assert load1.loadtype == 'W'
            assert load1.name == 'id'
            assert pytest.approx(load1.force, 0.0000001) == force1 + force2
            assert pytest.approx(load1.moment, 0.0000001) == moment1 + moment2
            assert load1.reference == ref

    # add generic load
    force1 = np.random.random(3) * 200 - 100
    moment1 = np.random.random(3) * 200 - 100
    ref1 = ReferenceFrame()
    ref1.shift(1, 2, 3)
    load1 = Load.fromarray('D', name='test', force=force1, moment=moment1, reference=ref1)

    force2 = np.random.random(3) * 200 - 100
    moment2 = np.random.random(3) * 200 - 100
    ref2 = ReferenceFrame()
    ref2.shift(-9, 0, 2)
    ref2.rotate_along([0, 2, 3], 47)
    load2 = GenericLoad.fromarray(force=force2, moment=moment2, reference=ref2)
    load2copy = copy.deepcopy(load2)
    load2copy.to_reference(load1.reference)
    
    load1 += load2

    assert load1.loadtype == 'D'
    assert load1.name == 'test'
    assert pytest.approx(load1.force, 0.0000001) == force1 + load2copy.force
    assert pytest.approx(load1.moment, 0.0000001) == moment1 + load2copy.moment
    assert load1.reference == ref1

    assert pytest.approx(load2.force, 0.0000001) == force2
    assert pytest.approx(load2.moment, 0.0000001) == moment2
    assert load2.reference == ref2

    # changes in load 2 dont affect load 1
    load2.shift(1, 2, 3)
    load2.xrotate(-24)

    assert load1.loadtype == 'D'
    assert load1.name == 'test'
    assert pytest.approx(load1.force, 0.0000001) == force1 + load2copy.force
    assert pytest.approx(load1.moment, 0.0000001) == moment1 + load2copy.moment
    assert load1.reference == ref1
    

def test_iadd_same_ref():
    '''Test __iadd__ method from Load class for same reference
    system
    '''

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for _ in range(100):
            force1 = np.random.random(3) * 200 - 100
            moment1 = np.random.random(3) * 200 - 100
            load1 = Load.fromarray('W', name='id', force=force1, moment=moment1, reference=ref)

            force2 = np.random.random(3) * 200 - 100
            moment2 = np.random.random(3) * 200 - 100
            load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref)

            load1 += load2

            # no changes in original loads
            assert load2.loadtype == 'D'
            assert load2.name == 'test'
            assert pytest.approx(load2.force, 0.0000001) == force2
            assert pytest.approx(load2.moment, 0.0000001) == moment2
            assert load2.reference == ref

            # new load is the sum
            assert load1.loadtype == 'W'
            assert load1.name == 'id'
            assert pytest.approx(load1.force, 0.0000001) == force1 + force2
            assert pytest.approx(load1.moment, 0.0000001) == moment1 + moment2
            assert load1.reference == ref

            # changes in the original laods do not affect the sum afterwards
            load2.shift(3, 1, 2)
            load2.xrotate(14)
            load2.yrotate(-34)

            assert load1.loadtype == 'W'
            assert load1.name == 'id'
            assert pytest.approx(load1.force, 0.0000001) == force1 + force2
            assert pytest.approx(load1.moment, 0.0000001) == moment1 + moment2
            assert load1.reference == ref


def test_iadd_different_ref():
    '''Test __iadd__ method from Load class for different reference
    systems
    '''

    coord = [-84.2, -0.4, 0, 3, 18.2]
    thetas = [-180, -134.5, -27, 0, 45, 90, 105.3]

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for x0, y0, z0 in itertools.product(coord, coord, coord):
            for lat, lon, theta in itertools.product(thetas, thetas, thetas): 
                direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                                    np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                                    np.sin(np.radians(lat))])

                ref_load2 = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                ref_load2.rotate_along(direction, theta)

                force1 = np.random.random(3) * 200 - 100
                moment1 = np.random.random(3) * 200 - 100
                load1 = Load.fromarray('W', name='id', force=force1, moment=moment1, reference=ref)

                force2 = np.random.random(3) * 200 - 100
                moment2 = np.random.random(3) * 200 - 100
                load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref_load2)

                load1 += load2

                load2_ref = copy.deepcopy(load2)
                load2_ref.to_reference(ref)

                # no changes in original loads
                assert load2.loadtype == 'D'
                assert load2.name == 'test'
                assert pytest.approx(load2.force, 0.0000001) == force2
                assert pytest.approx(load2.moment, 0.0000001) == moment2
                assert load2.reference == ref_load2

                # new load is the sum
                assert load1.loadtype == 'W'
                assert load1.name == 'id'
                assert pytest.approx(load1.force, 0.0000001) == force1 + load2_ref.force
                assert pytest.approx(load1.moment, 0.0000001) == moment1 + load2_ref.moment
                assert load1.reference == ref

                # changes in the original laods do not affect the sum afterwards
                load2.shift(3, 1, 2)
                load2.xrotate(14)
                load2.yrotate(-34)

                assert load1.loadtype == 'W'
                assert load1.name == 'id'
                assert pytest.approx(load1.force, 0.0000001) == force1 + load2_ref.force
                assert pytest.approx(load1.moment, 0.0000001) == moment1 + load2_ref.moment
                assert load1.reference == ref


def test_isub_inputs():
    '''Test __isub__ method from Load class for different inputs
    '''

    # nan handling
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        force1 = np.random.random(3) * 200 - 100
        moment1 = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('W', name='id', force=force1, moment=moment1, reference=ref)

        force2 = [np.nan, np.nan, np.nan]
        moment2 = [np.nan, np.nan, np.nan]
        load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref)
        load1 -= load2
        assert load1.loadtype == 'W'
        assert load1.name == 'id'
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert all(np.isnan(load1.force))
        assert all(np.isnan(load1.moment))
        assert load1.reference == ref


        force2 = np.random.random(3)
        moment2 = [np.nan, np.nan, np.nan]
        load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref)
        load1 -= load2
        assert load1.loadtype == 'W'
        assert load1.name == 'id'
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert all(np.isnan(load1.force))
        assert all(np.isnan(load1.moment))
        assert load1.reference == ref


        force2 = [np.nan, np.nan, np.nan]
        moment2 = np.random.random(3)
        load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref)
        load1 -= load2
        assert load1.loadtype == 'W'
        assert load1.name == 'id'
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert all(np.isnan(load1.force))
        assert all(np.isnan(load1.moment))
        assert load1.reference == ref


        force2 = [np.nan, 3, 4]
        moment2 = np.random.random(3)
        load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref)
        load1 -= load2
        assert load1.loadtype == 'W'
        assert load1.name == 'id'
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert all(np.isnan(load1.force))
        assert all(np.isnan(load1.moment))
        assert load1.reference == ref

        force2 = [np.nan, 3, np.nan]
        moment2 = np.random.random(3)
        load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref)
        load1 -= load2
        assert load1.loadtype == 'W'
        assert load1.name == 'id'
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert all(np.isnan(load1.force))
        assert all(np.isnan(load1.moment))
        assert load1.reference == ref

        force2 = np.random.random(3)
        moment2 = [np.nan, 3, 4]
        load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref)
        load1 -= load2
        assert load1.loadtype == 'W'
        assert load1.name == 'id'
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert all(np.isnan(load1.force))
        assert all(np.isnan(load1.moment))
        assert load1.reference == ref

        force2 = np.random.random(3)
        moment2 = [np.nan, np.nan, 4]
        load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref)
        load1 -= load2
        assert load1.loadtype == 'W'
        assert load1.name == 'id'
        assert load2.loadtype == 'D'
        assert load2.name == 'test'
        assert all(np.isnan(load1.force))
        assert all(np.isnan(load1.moment))
        assert load1.reference == ref

    # wrong input type
    with pytest.raises(TypeError):
        force1 = np.random.random(3) * 200 - 100
        moment1 = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('D', name='test', force=force1, moment=moment1, reference=ref)
        load1 -= 1
    
    with pytest.raises(TypeError):
        force1 = np.random.random(3) * 200 - 100
        moment1 = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('D', name='test', force=force1, moment=moment1, reference=ref)
        load1 -= [1, 2, 3]
    
    with pytest.raises(TypeError):
        force1 = np.random.random(3) * 200 - 100
        moment1 = np.random.random(3) * 200 - 100
        load1 = Load.fromarray('D', name='test', force=force1, moment=moment1, reference=ref)
        load1 -= 'load'

    # different data types
    force = [[1, 2, 3], np.array([1, 2, 3]), [1.23, 2.42, 3.12], np.array([1.23, 2.42, 3.12])]
    moment = [[1, 2, 3], np.array([1, 2, 3]), [1.23, 2.42, 3.12], np.array([1.23, 2.42, 3.12])]
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    refs = [ref1, ref2, ref3]

    for force1, force2, moment1, moment2 in itertools.product(force, force, moment, moment):
        for ref in refs:
    
            load1 = Load.fromarray('W', name='id', force=force1, moment=moment1, reference=ref)
            load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref)

            load1 -= load2

            force1 = np.array(force1)
            force2 = np.array(force2)

            moment1 = np.array(moment1)
            moment2 = np.array(moment2)

            assert load2.loadtype == 'D'
            assert load2.name == 'test'
            assert pytest.approx(load2.force, 0.0000001) == force2
            assert pytest.approx(load2.moment, 0.0000001) == moment2
            assert load2.reference == ref

            # new load is the sum
            assert load1.loadtype == 'W'
            assert load1.name == 'id'
            assert pytest.approx(load1.force, 0.0000001) == force1 - force2
            assert pytest.approx(load1.moment, 0.0000001) == moment1 - moment2
            assert load1.reference == ref
    

    vals = [0, 1, -1.2323]
    for xforce1, yforce1, zforce1, xmoment1, ymoment1, zmoment1 in itertools.product(vals, vals, vals, vals, vals, vals):
        for xforce2, yforce2, zforce2, xmoment2, ymoment2, zmoment2 in itertools.product(vals, vals, vals, vals, vals, vals):

            load1 = Load('W', name='id', xforce=xforce1, yforce=yforce1, zforce=zforce1,
                                xmoment=xmoment1, ymoment=ymoment1, zmoment=zmoment1, reference=ref)
            load2 = Load('D', name='test', xforce=xforce2, yforce=yforce2, zforce=zforce2,
                                xmoment=xmoment2, ymoment=ymoment2, zmoment=zmoment2, reference=ref)

            load1 -= load2

            force1 = np.array([xforce1, yforce1, zforce1])
            force2 = np.array([xforce2, yforce2, zforce2])

            moment1 = np.array([xmoment1, ymoment1, zmoment1])
            moment2 = np.array([xmoment2, ymoment2, zmoment2])

            # no changes in original loads
            assert load2.loadtype == 'D'
            assert load2.name == 'test'
            assert pytest.approx(load2.force, 0.0000001) == force2
            assert pytest.approx(load2.moment, 0.0000001) == moment2
            assert load2.reference == ref

            # new load is the sum
            assert load1.loadtype == 'W'
            assert load1.name == 'id'
            assert pytest.approx(load1.force, 0.0000001) == force1 - force2
            assert pytest.approx(load1.moment, 0.0000001) == moment1 - moment2
            assert load1.reference == ref


    # add generic load
    force1 = np.random.random(3) * 200 - 100
    moment1 = np.random.random(3) * 200 - 100
    ref1 = ReferenceFrame()
    ref1.shift(1, 2, 3)
    load1 = Load.fromarray('D', name='test', force=force1, moment=moment1, reference=ref1)

    force2 = np.random.random(3) * 200 - 100
    moment2 = np.random.random(3) * 200 - 100
    ref2 = ReferenceFrame()
    ref2.shift(-9, 0, 2)
    ref2.rotate_along([0, 2, 3], 47)
    load2 = GenericLoad.fromarray(force=force2, moment=moment2, reference=ref2)
    load2copy = copy.deepcopy(load2)
    load2copy.to_reference(load1.reference)
    
    load1 -= load2

    assert load1.loadtype == 'D'
    assert load1.name == 'test'
    assert pytest.approx(load1.force, 0.0000001) == force1 - load2copy.force
    assert pytest.approx(load1.moment, 0.0000001) == moment1 - load2copy.moment
    assert load1.reference == ref1

    assert pytest.approx(load2.force, 0.0000001) == force2
    assert pytest.approx(load2.moment, 0.0000001) == moment2
    assert load2.reference == ref2

    # changes in load 2 dont affect load 1
    load2.shift(1, 2, 3)
    load2.xrotate(-24)

    assert load1.loadtype == 'D'
    assert load1.name == 'test'
    assert pytest.approx(load1.force, 0.0000001) == force1 - load2copy.force
    assert pytest.approx(load1.moment, 0.0000001) == moment1 - load2copy.moment
    assert load1.reference == ref1
    

def test_isub_same_ref():
    '''Test __isub__ method from Load class for same reference
    system
    '''

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for _ in range(100):
            force1 = np.random.random(3) * 200 - 100
            moment1 = np.random.random(3) * 200 - 100
            load1 = Load.fromarray('W', name='id', force=force1, moment=moment1, reference=ref)

            force2 = np.random.random(3) * 200 - 100
            moment2 = np.random.random(3) * 200 - 100
            load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref)

            load1 -= load2

            # no changes in original loads
            assert load2.loadtype == 'D'
            assert load2.name == 'test'
            assert pytest.approx(load2.force, 0.0000001) == force2
            assert pytest.approx(load2.moment, 0.0000001) == moment2
            assert load2.reference == ref

            # new load is the sum
            assert load1.loadtype == 'W'
            assert load1.name == 'id'
            assert pytest.approx(load1.force, 0.0000001) == force1 - force2
            assert pytest.approx(load1.moment, 0.0000001) == moment1 - moment2
            assert load1.reference == ref

            # changes in the original laods do not affect the sum afterwards
            load2.shift(3, 1, 2)
            load2.xrotate(14)
            load2.yrotate(-34)

            assert load1.loadtype == 'W'
            assert load1.name == 'id'
            assert pytest.approx(load1.force, 0.0000001) == force1 - force2
            assert pytest.approx(load1.moment, 0.0000001) == moment1 - moment2
            assert load1.reference == ref


def test_isub_different_ref():
    '''Test __isub__ method from Load class for different reference
    systems
    '''

    coord = [-84.2, -0.4, 0, 3, 18.2]
    thetas = [-180, -134.5, -27, 0, 45, 90, 105.3]

    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for x0, y0, z0 in itertools.product(coord, coord, coord):
            for lat, lon, theta in itertools.product(thetas, thetas, thetas): 
                direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                                    np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                                    np.sin(np.radians(lat))])

                ref_load2 = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                ref_load2.rotate_along(direction, theta)

                force1 = np.random.random(3) * 200 - 100
                moment1 = np.random.random(3) * 200 - 100
                load1 = Load.fromarray('W', name='id', force=force1, moment=moment1, reference=ref)

                force2 = np.random.random(3) * 200 - 100
                moment2 = np.random.random(3) * 200 - 100
                load2 = Load.fromarray('D', name='test', force=force2, moment=moment2, reference=ref_load2)

                load1 -= load2

                load2_ref = copy.deepcopy(load2)
                load2_ref.to_reference(ref)

                # no changes in original loads
                assert load2.loadtype == 'D'
                assert load2.name == 'test'
                assert pytest.approx(load2.force, 0.0000001) == force2
                assert pytest.approx(load2.moment, 0.0000001) == moment2
                assert load2.reference == ref_load2

                # new load is the sum
                assert load1.loadtype == 'W'
                assert load1.name == 'id'
                assert pytest.approx(load1.force, 0.0000001) == force1 - load2_ref.force
                assert pytest.approx(load1.moment, 0.0000001) == moment1 - load2_ref.moment
                assert load1.reference == ref

                # changes in the original laods do not affect the sum afterwards
                load2.shift(3, 1, 2)
                load2.xrotate(14)
                load2.yrotate(-34)

                assert load1.loadtype == 'W'
                assert load1.name == 'id'
                assert pytest.approx(load1.force, 0.0000001) == force1 - load2_ref.force
                assert pytest.approx(load1.moment, 0.0000001) == moment1 - load2_ref.moment
                assert load1.reference == ref


def test_imul():
    '''Test __imul__ method from Load class'''

    # test nan
    force = [1, 2, 3]
    moment = [3, 4, 5]
    load = Load.fromarray('D', name='test', force=force, moment=moment)
    load *= np.nan
    assert all(np.isnan(load.force))
    assert all(np.isnan(load.moment))
    assert load.reference == ReferenceFrame()

    # test wrong data type
    with pytest.raises(TypeError):
        load *= '3'

    with pytest.raises(TypeError):
        load *= [3]

    with pytest.raises(TypeError):
        load *= np.array([3])


    vals = [-123, 0, 0.5, 1, 101235]
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for xforce, yforce, zforce, xmoment, ymoment, zmoment in itertools.product(vals, vals, vals, vals, vals, vals):
            for multby in vals:
                load = Load('D', name='test', xforce=xforce, yforce=yforce, zforce=zforce,
                                    xmoment=xmoment, ymoment=ymoment, zmoment=zmoment,
                                    reference=ref)
                load *= multby

                force = np.array([xforce, yforce, zforce])
                moment = np.array([xmoment, ymoment, zmoment])

                assert load.loadtype == 'D'
                assert load.name == 'test'
                assert pytest.approx(load.force, 0.0000001) == force * multby
                assert pytest.approx(load.moment, 0.0000001) == moment * multby
                assert load.reference == ref


def test_itruediv():
    '''Test __itruediv__ method from Load class'''

    # test nan
    force = [1, 2, 3]
    moment = [3, 4, 5]
    load = Load.fromarray('D', name='test', force=force, moment=moment)

    load /= np.nan
    assert all(np.isnan(load.force))
    assert all(np.isnan(load.moment))
    assert load.reference == ReferenceFrame()

    # test wrong data type
    with pytest.raises(TypeError):
        load /= '3'

    with pytest.raises(TypeError):
        load /= [3]
    
    with pytest.raises(TypeError):
        load /= np.array([3])

    # test division by zero
    with pytest.raises(ValueError):
        load /= 0

    vals = [-123, 0, 0.5, 1, 101235]
    divbys = [-123, 0.5, 1, 101235]
    ref1 = ReferenceFrame()
    ref2 = copy.deepcopy(ref1)
    ref2.rotate_along([1, 2, 3], 123)
    ref3 = copy.deepcopy(ref1)
    ref3.shift(3, 2, 1)
    for ref in [ref1, ref2, ref3]:
        for xforce, yforce, zforce, xmoment, ymoment, zmoment in itertools.product(vals, vals, vals, vals, vals, vals):
            for divby in divbys:
                load1 = Load('D', name='test', xforce=xforce, yforce=yforce, zforce=zforce,
                                    xmoment=xmoment, ymoment=ymoment, zmoment=zmoment,
                                    reference=ref)
                load1 /= divby

                force = np.array([xforce, yforce, zforce])
                moment = np.array([xmoment, ymoment, zmoment])

                assert load.loadtype == 'D'
                assert load.name == 'test'
                assert pytest.approx(load1.force, 0.0000001) == force / divby
                assert pytest.approx(load1.moment, 0.0000001) == moment / divby
                assert load1.reference == ref