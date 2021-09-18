'''Module for testing the ReferenceFrame class in
cirsoc_402.loads.referenceframe
'''

import numpy as np
import pytest
import itertools

from cirsoc_402.load import ReferenceFrame


def test_init():
    '''Test __init__ method from ReferenceFrame class
    '''
    vals = [-123, 0, 0.5, 1, 101235]
    for x, y, z in itertools.product(vals, vals, vals):
        frame = ReferenceFrame(xcoord=x, ycoord=y, zcoord=z)
        assert all(frame.origin == np.array([x, y, z]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert all(frame.zversor == np.array([0, 0, 1]))
    
    for x, y in itertools.product(vals, vals):
        frame = ReferenceFrame(xcoord=x, ycoord=y)
        assert all(frame.origin == np.array([x, y, 0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(xcoord=x, zcoord=y)
        assert all(frame.origin == np.array([x, 0, y]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(ycoord=x, zcoord=y)
        assert all(frame.origin == np.array([0, x, y]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert all(frame.zversor == np.array([0, 0, 1]))
    
    for x in vals:
        frame = ReferenceFrame(xcoord=x)
        assert all(frame.origin == np.array([x, 0, 0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(ycoord=x)
        assert all(frame.origin == np.array([0, x, 0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(zcoord=x)
        assert all(frame.origin == np.array([0, 0, x]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert all(frame.zversor == np.array([0, 0, 1]))
    
    # check nan behavior
    frame = ReferenceFrame(xcoord=np.nan)
    assert np.isnan(frame.origin[0]) == True
    assert frame.origin[1] == 0
    assert frame.origin[2] == 0
    assert all(frame.xversor == np.array([1, 0, 0]))
    assert all(frame.yversor == np.array([0, 1, 0]))
    assert all(frame.zversor == np.array([0, 0, 1]))

    frame = ReferenceFrame(ycoord=np.nan)
    assert frame.origin[0] == 0
    assert np.isnan(frame.origin[1]) == True
    assert frame.origin[2] == 0
    assert all(frame.xversor == np.array([1, 0, 0]))
    assert all(frame.yversor == np.array([0, 1, 0]))
    assert all(frame.zversor == np.array([0, 0, 1]))

    frame = ReferenceFrame(zcoord=np.nan)
    assert frame.origin[0] == 0
    assert frame.origin[1] == 0
    assert np.isnan(frame.origin[2]) == True
    assert all(frame.xversor == np.array([1, 0, 0]))
    assert all(frame.yversor == np.array([0, 1, 0]))
    assert all(frame.zversor == np.array([0, 0, 1]))

    frame = ReferenceFrame(xcoord=np.nan, ycoord=np.nan)
    assert np.isnan(frame.origin[0]) == True
    assert np.isnan(frame.origin[1]) == True
    assert frame.origin[2] == 0
    assert all(frame.xversor == np.array([1, 0, 0]))
    assert all(frame.yversor == np.array([0, 1, 0]))
    assert all(frame.zversor == np.array([0, 0, 1]))

    frame = ReferenceFrame(xcoord=np.nan, zcoord=np.nan)
    assert np.isnan(frame.origin[0]) == True
    assert frame.origin[1] == 0
    assert np.isnan(frame.origin[2]) == True
    assert all(frame.xversor == np.array([1, 0, 0]))
    assert all(frame.yversor == np.array([0, 1, 0]))
    assert all(frame.zversor == np.array([0, 0, 1]))

    frame = ReferenceFrame(ycoord=np.nan, zcoord=np.nan)
    assert frame.origin[0] == 0
    assert np.isnan(frame.origin[1]) == True
    assert np.isnan(frame.origin[2]) == True
    assert all(frame.xversor == np.array([1, 0, 0]))
    assert all(frame.yversor == np.array([0, 1, 0]))
    assert all(frame.zversor == np.array([0, 0, 1]))

    frame = ReferenceFrame(xcoord=np.nan, ycoord=np.nan, zcoord=np.nan)
    assert np.isnan(frame.origin[0]) == True
    assert np.isnan(frame.origin[1]) == True
    assert np.isnan(frame.origin[2]) == True
    assert all(frame.xversor == np.array([1, 0, 0]))
    assert all(frame.yversor == np.array([0, 1, 0]))
    assert all(frame.zversor == np.array([0, 0, 1]))
    
    # check type error
    with pytest.raises(TypeError):
        frame = ReferenceFrame(xcoord=[1])
    
    with pytest.raises(TypeError):
        frame = ReferenceFrame(xcoord='1')
    
    with pytest.raises(TypeError):
        frame = ReferenceFrame(ycoord=[1])
    
    with pytest.raises(TypeError):
        frame = ReferenceFrame(ycoord='1')
    
    with pytest.raises(TypeError):
        frame = ReferenceFrame(zcoord=[1])
    
    with pytest.raises(TypeError):
        frame = ReferenceFrame(zcoord='1')


def test_repr():
    '''Test __repr__ method from ReferenceFrame class
    '''

    frame = ReferenceFrame(xcoord=1, ycoord=2, zcoord=3)
    txt = "R = ({:.2f}, {:.2f}, {:.2f})\n".format(1, 2, 3)
    txt += "ex = ({:.2f}, {:.2f}, {:.2f})\n".format(1, 0, 0)
    txt += "ey = ({:.2f}, {:.2f}, {:.2f})\n".format(0, 1, 0)
    txt += "ez = ({:.2f}, {:.2f}, {:.2f})".format(0, 0, 1)
    assert frame.__repr__() == txt

    frame = ReferenceFrame()
    txt = "R = ({:.2f}, {:.2f}, {:.2f})\n".format(0, 0, 0)
    txt += "ex = ({:.2f}, {:.2f}, {:.2f})\n".format(1, 0, 0)
    txt += "ey = ({:.2f}, {:.2f}, {:.2f})\n".format(0, 1, 0)
    txt += "ez = ({:.2f}, {:.2f}, {:.2f})".format(0, 0, 1)
    assert frame.__repr__() == txt

    frame = ReferenceFrame()
    frame.xshift(2.5)
    txt = "R = ({:.2f}, {:.2f}, {:.2f})\n".format(2.5, 0, 0)
    txt += "ex = ({:.2f}, {:.2f}, {:.2f})\n".format(1, 0, 0)
    txt += "ey = ({:.2f}, {:.2f}, {:.2f})\n".format(0, 1, 0)
    txt += "ez = ({:.2f}, {:.2f}, {:.2f})".format(0, 0, 1)
    assert frame.__repr__() == txt

    frame = ReferenceFrame()
    frame.xshift(2.5)
    frame.zshift(-4.5)
    txt = "R = ({:.2f}, {:.2f}, {:.2f})\n".format(2.5, 0, -4.5)
    txt += "ex = ({:.2f}, {:.2f}, {:.2f})\n".format(1, 0, 0)
    txt += "ey = ({:.2f}, {:.2f}, {:.2f})\n".format(0, 1, 0)
    txt += "ez = ({:.2f}, {:.2f}, {:.2f})".format(0, 0, 1)
    assert frame.__repr__() == txt

    frame = ReferenceFrame()
    frame.xshift(2.5)
    frame.yshift(33)
    frame.zshift(-4.5)
    txt = "R = ({:.2f}, {:.2f}, {:.2f})\n".format(2.5, 33, -4.5)
    txt += "ex = ({:.2f}, {:.2f}, {:.2f})\n".format(1, 0, 0)
    txt += "ey = ({:.2f}, {:.2f}, {:.2f})\n".format(0, 1, 0)
    txt += "ez = ({:.2f}, {:.2f}, {:.2f})".format(0, 0, 1)
    assert frame.__repr__() == txt

    frame = ReferenceFrame()
    frame.xshift_ref(2.5)
    txt = "R = ({:.2f}, {:.2f}, {:.2f})\n".format(2.5, 0, 0)
    txt += "ex = ({:.2f}, {:.2f}, {:.2f})\n".format(1, 0, 0)
    txt += "ey = ({:.2f}, {:.2f}, {:.2f})\n".format(0, 1, 0)
    txt += "ez = ({:.2f}, {:.2f}, {:.2f})".format(0, 0, 1)
    assert frame.__repr__() == txt

    frame = ReferenceFrame()
    frame.xshift_ref(2.5)
    frame.zshift_ref(-4.5)
    txt = "R = ({:.2f}, {:.2f}, {:.2f})\n".format(2.5, 0, -4.5)
    txt += "ex = ({:.2f}, {:.2f}, {:.2f})\n".format(1, 0, 0)
    txt += "ey = ({:.2f}, {:.2f}, {:.2f})\n".format(0, 1, 0)
    txt += "ez = ({:.2f}, {:.2f}, {:.2f})".format(0, 0, 1)
    assert frame.__repr__() == txt

    frame = ReferenceFrame()
    frame.xshift_ref(2.5)
    frame.yshift_ref(33)
    frame.zshift_ref(-4.5)
    txt = "R = ({:.2f}, {:.2f}, {:.2f})\n".format(2.5, 33, -4.5)
    txt += "ex = ({:.2f}, {:.2f}, {:.2f})\n".format(1, 0, 0)
    txt += "ey = ({:.2f}, {:.2f}, {:.2f})\n".format(0, 1, 0)
    txt += "ez = ({:.2f}, {:.2f}, {:.2f})".format(0, 0, 1)
    assert frame.__repr__() == txt

    frame = ReferenceFrame(xcoord=1, ycoord=2, zcoord=3)
    frame.xrotate(90)
    txt = "R = ({:.2f}, {:.2f}, {:.2f})\n".format(1, 2, 3)
    txt += "ex = ({:.2f}, {:.2f}, {:.2f})\n".format(1, 0, 0)
    txt += "ey = ({:.2f}, {:.2f}, {:.2f})\n".format(0, 0, 1)
    txt += "ez = ({:.2f}, {:.2f}, {:.2f})".format(0, -1, 0)
    assert frame.__repr__() == txt

    frame = ReferenceFrame(xcoord=1, ycoord=2, zcoord=3)
    frame.yrotate(-90)
    txt = "R = ({:.2f}, {:.2f}, {:.2f})\n".format(1, 2, 3)
    txt += "ex = ({:.2f}, {:.2f}, {:.2f})\n".format(0, 0, 1)
    txt += "ey = ({:.2f}, {:.2f}, {:.2f})\n".format(0, 1, 0)
    txt += "ez = ({:.2f}, {:.2f}, {:.2f})".format(-1, 0, 0)
    assert frame.__repr__() == txt

    frame = ReferenceFrame(xcoord=1, ycoord=2, zcoord=3)
    frame.zrotate(90)
    txt = "R = ({:.2f}, {:.2f}, {:.2f})\n".format(1, 2, 3)
    txt += "ex = ({:.2f}, {:.2f}, {:.2f})\n".format(0, 1, 0)
    txt += "ey = ({:.2f}, {:.2f}, {:.2f})\n".format(-1, 0, 0)
    txt += "ez = ({:.2f}, {:.2f}, {:.2f})".format(0, 0, 1)
    assert frame.__repr__() == txt


    frame = ReferenceFrame(xcoord=1, ycoord=2, zcoord=3)
    frame.zrotate(45)
    txt = "R = ({:.2f}, {:.2f}, {:.2f})\n".format(1, 2, 3)
    txt += "ex = ({:.2f}, {:.2f}, {:.2f})\n".format(1/np.sqrt(2), 1/np.sqrt(2), 0)
    txt += "ey = ({:.2f}, {:.2f}, {:.2f})\n".format(-1/np.sqrt(2), 1/np.sqrt(2), 0)
    txt += "ez = ({:.2f}, {:.2f}, {:.2f})".format(0, 0, 1)
    assert frame.__repr__() == txt


def test_eq():
    '''Test __eq__ method from ReferenceFrame class
    '''

    vals = [-123, 0, 0.5, 1, 101235]
    for x, y, z in itertools.product(vals, vals, vals):
        frame1 = ReferenceFrame(xcoord=x, ycoord=y, zcoord=z)
        frame2 = ReferenceFrame(xcoord=x, ycoord=y, zcoord=z)
        assert(frame1 ==  frame2)

        direction = np.random.random(3)
        rotation = np.random.random() * 360 - 180
        frame1.rotate_along(direction, rotation)
        frame2.rotate_along(direction, rotation)
        displacement = np.random.random(3) * 10 - 5
        frame1.shift(displacement[0], displacement[1], displacement[2])
        frame2.shift(displacement[0], displacement[1], displacement[2])
        assert(frame1 ==  frame2)

        frame1 = ReferenceFrame(xcoord=np.nan, ycoord=y, zcoord=z)
        frame2 = ReferenceFrame(xcoord=x, ycoord=y, zcoord=z)
        assert not(frame1 ==  frame2)

        frame1 = ReferenceFrame(xcoord=x, ycoord=y, zcoord=z)
        frame2 = ReferenceFrame(xcoord=np.nan, ycoord=y, zcoord=z)
        assert not(frame1 ==  frame2)

        frame1 = ReferenceFrame(xcoord=np.nan, ycoord=y, zcoord=z)
        frame2 = ReferenceFrame(xcoord=np.nan, ycoord=y, zcoord=z)
        assert not(frame1 ==  frame2)

        frame1 = ReferenceFrame(xcoord=np.nan, ycoord=y, zcoord=z)
        frame2 = ReferenceFrame(xcoord=np.nan, ycoord=np.nan, zcoord=z)
        assert not(frame1 ==  frame2)

        frame1 = ReferenceFrame(xcoord=np.nan, ycoord=np.nan, zcoord=np.nan)
        frame2 = ReferenceFrame(xcoord=np.nan, ycoord=np.nan, zcoord=np.nan)
        assert not(frame1 ==  frame2)


    vals1 = [-123, 0, 0.5, 1, 101235]
    vals2 = [-122, 0.3, 3.5, 4.6, 1235]
    for x1, y1, z1 in itertools.product(vals1, vals1, vals1):
        frame1 = ReferenceFrame(xcoord=x1, ycoord=y1, zcoord=z1)
        for x2, y2, z2 in itertools.product(vals2, vals2, vals2):
            frame2 = ReferenceFrame(xcoord=x2, ycoord=y2, zcoord=z2)
            assert not(frame1 ==  frame2)


def test_shift():
    '''Test shift method from ReferenceFrame class
    '''

    vals = [-123, 0, 0.5, 23, 56.91]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for x, y, z in itertools.product(vals, vals, vals):
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            frame.shift(x, y, z)
            assert all(frame.origin == np.array([x + x0, y + y0, z + z0]))
            assert all(frame.xversor == np.array([1, 0, 0]))
            assert all(frame.yversor == np.array([0, 1, 0]))
            assert all(frame.zversor == np.array([0, 0, 1]))

    theta = [-35, 23.5, 0]
    vals = [-123, 0, 3.5]
    for thetax, thetay, thetaz in itertools.product(theta, theta, theta):
        for x0, y0, z0 in itertools.product(vals, vals, vals):
            for x, y, z in itertools.product(vals, vals, vals):
                frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                frame.xrotate(thetax)
                frame.yrotate(thetay)
                frame.zrotate(thetaz)
                frame.shift(x, y, z)
                assert all(frame.origin == np.array([x + x0, y + y0, z + z0]))
    
    theta = [-35, 23.5, 0]
    vals = [-123, 0, 3.5]
    for thetax, thetay, thetaz in itertools.product(theta, theta, theta):
        for x0, y0, z0 in itertools.product(vals, vals, vals):
            for x, y, z in itertools.product(vals, vals, vals):
                frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                frame.xrotate_ref(thetax)
                frame.yrotate_ref(thetay)
                frame.zrotate_ref(thetaz)
                frame.shift(x, y, z)
                assert all(frame.origin == np.array([x + x0, y + y0, z + z0]))


def test_xshift():
    '''Test xshift method from ReferenceFrame class
    '''

    vals = [-123, 0, 0.5, 23, 56.91]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for shift in vals:
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            frame.xshift(shift)
            assert all(frame.origin == np.array([shift + x0, y0, z0]))
            assert all(frame.xversor == np.array([1, 0, 0]))
            assert all(frame.yversor == np.array([0, 1, 0]))
            assert all(frame.zversor == np.array([0, 0, 1]))

    theta = [-35, 23.5, 0]
    vals = [-123, 0, 3.5]
    for thetax, thetay, thetaz in itertools.product(theta, theta, theta):
        for x0, y0, z0 in itertools.product(vals, vals, vals):
            for shift in vals:
                frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                frame.xrotate(thetax)
                frame.yrotate(thetay)
                frame.zrotate(thetaz)
                frame.xshift(shift)
                assert all(frame.origin == np.array([shift + x0, y0, z0]))
    
    theta = [-35, 23.5, 0]
    vals = [-123, 0, 3.5]
    for thetax, thetay, thetaz in itertools.product(theta, theta, theta):
        for x0, y0, z0 in itertools.product(vals, vals, vals):
            for shift in vals:
                frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                frame.xrotate_ref(thetax)
                frame.yrotate_ref(thetay)
                frame.zrotate_ref(thetaz)
                frame.xshift(shift)
                assert all(frame.origin == np.array([shift + x0, y0, z0]))


def test_yshift():
    '''Test yshift method from ReferenceFrame class
    '''

    vals = [-123, 0, 0.5, 23, 56.91]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for shift in vals:
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            frame.yshift(shift)
            assert all(frame.origin == np.array([x0, shift + y0, z0]))
            assert all(frame.xversor == np.array([1, 0, 0]))
            assert all(frame.yversor == np.array([0, 1, 0]))
            assert all(frame.zversor == np.array([0, 0, 1]))

    theta = [-35, 23.5, 0]
    vals = [-123, 0, 3.5]
    for thetax, thetay, thetaz in itertools.product(theta, theta, theta):
        for x0, y0, z0 in itertools.product(vals, vals, vals):
            for shift in vals:
                frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                frame.xrotate(thetax)
                frame.yrotate(thetay)
                frame.zrotate(thetaz)
                frame.yshift(shift)
                assert all(frame.origin == np.array([x0, shift + y0, z0]))
    
    theta = [-35, 23.5, 0]
    vals = [-123, 0, 3.5]
    for thetax, thetay, thetaz in itertools.product(theta, theta, theta):
        for x0, y0, z0 in itertools.product(vals, vals, vals):
            for shift in vals:
                frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                frame.xrotate_ref(thetax)
                frame.yrotate_ref(thetay)
                frame.zrotate_ref(thetaz)
                frame.yshift(shift)
                assert all(frame.origin == np.array([x0, shift + y0, z0]))


def test_zshift():
    '''Test zshift method from ReferenceFrame class
    '''

    vals = [-123, 0, 0.5, 23, 56.91]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for shift in vals:
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            frame.zshift(shift)
            assert all(frame.origin == np.array([x0, y0, shift + z0]))
            assert all(frame.xversor == np.array([1, 0, 0]))
            assert all(frame.yversor == np.array([0, 1, 0]))
            assert all(frame.zversor == np.array([0, 0, 1]))

    theta = [-35, 23.5, 0]
    vals = [-123, 0, 3.5]
    for thetax, thetay, thetaz in itertools.product(theta, theta, theta):
        for x0, y0, z0 in itertools.product(vals, vals, vals):
            for shift in vals:
                frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                frame.xrotate(thetax)
                frame.yrotate(thetay)
                frame.zrotate(thetaz)
                frame.zshift(shift)
                assert all(frame.origin == np.array([x0, y0, shift + z0]))
    
    theta = [-35, 23.5, 0]
    vals = [-123, 0, 3.5]
    for thetax, thetay, thetaz in itertools.product(theta, theta, theta):
        for x0, y0, z0 in itertools.product(vals, vals, vals):
            for shift in vals:
                frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                frame.xrotate_ref(thetax)
                frame.yrotate_ref(thetay)
                frame.zrotate_ref(thetaz)
                frame.zshift(shift)
                assert all(frame.origin == np.array([x0, y0, shift + z0]))


def test_shift_ref():
    '''Test shift_ref method from ReferenceFrame class
    '''

    vals = [-123, 0, 0.5, 23, 56.91]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for x, y, z in itertools.product(vals, vals, vals):
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            frame.shift_ref(x, y, z)
            assert all(frame.origin == np.array([x + x0, y + y0, z + z0]))
            assert all(frame.xversor == np.array([1, 0, 0]))
            assert all(frame.yversor == np.array([0, 1, 0]))
            assert all(frame.zversor == np.array([0, 0, 1]))

    theta = [-35, 23.5, 0]
    vals = [-123, 0, 3.5]
    for thetax, thetay, thetaz in itertools.product(theta, theta, theta):
        for x0, y0, z0 in itertools.product(vals, vals, vals):
            for x, y, z in itertools.product(vals, vals, vals):
                frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                frame.xrotate(thetax)
                frame.yrotate(thetay)
                frame.zrotate(thetaz)
                frame.shift_ref(x, y, z)

                xversor = frame.xversor
                yversor = frame.yversor
                zversor = frame.zversor
                movement = x * xversor + y * yversor + z * zversor
                assert all(frame.origin == np.array([x0, y0, z0]) + movement)
    
    theta = [-35, 23.5, 0]
    vals = [-123, 0, 3.5]
    for thetax, thetay, thetaz in itertools.product(theta, theta, theta):
        for x0, y0, z0 in itertools.product(vals, vals, vals):
            for x, y, z in itertools.product(vals, vals, vals):
                frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                frame.xrotate_ref(thetax)
                frame.yrotate_ref(thetay)
                frame.zrotate_ref(thetaz)
                frame.shift_ref(x, y, z)
                
                xversor = frame.xversor
                yversor = frame.yversor
                zversor = frame.zversor
                movement = x * xversor + y * yversor + z * zversor
                assert all(frame.origin == np.array([x0, y0, z0]) + movement)


def test_xshift_ref():
    '''Test xshift_ref method from ReferenceFrame class
    '''

    vals = [-123, 0, 0.5, 23, 56.91]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for shift in vals:
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            frame.xshift_ref(shift)
            assert all(frame.origin == np.array([x0 + shift, y0, z0]))
            assert all(frame.xversor == np.array([1, 0, 0]))
            assert all(frame.yversor == np.array([0, 1, 0]))
            assert all(frame.zversor == np.array([0, 0, 1]))

    theta = [-35, 23.5, 0]
    vals = [-123, 0, 3.5]
    for thetax, thetay, thetaz in itertools.product(theta, theta, theta):
        for x0, y0, z0 in itertools.product(vals, vals, vals):
            for shift in vals:
                frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                frame.xrotate(thetax)
                frame.yrotate(thetay)
                frame.zrotate(thetaz)
                frame.xshift_ref(shift)

                xversor = frame.xversor
                movement = shift * xversor
                assert all(frame.origin == np.array([x0, y0, z0]) + movement)
    
    theta = [-35, 23.5, 0]
    vals = [-123, 0, 3.5]
    for thetax, thetay, thetaz in itertools.product(theta, theta, theta):
        for x0, y0, z0 in itertools.product(vals, vals, vals):
            for shift in vals:
                frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                frame.xrotate_ref(thetax)
                frame.yrotate_ref(thetay)
                frame.zrotate_ref(thetaz)
                frame.xshift_ref(shift)

                xversor = frame.xversor
                movement = shift * xversor
                assert all(frame.origin == np.array([x0, y0, z0]) + movement)


def test_yshift_ref():
    '''Test yshift_ref method from ReferenceFrame class
    '''

    vals = [-123, 0, 0.5, 23, 56.91]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for shift in vals:
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            frame.yshift_ref(shift)
            assert all(frame.origin == np.array([x0, y0 + shift, z0]))
            assert all(frame.xversor == np.array([1, 0, 0]))
            assert all(frame.yversor == np.array([0, 1, 0]))
            assert all(frame.zversor == np.array([0, 0, 1]))

    theta = [-35, 23.5, 0]
    vals = [-123, 0, 3.5]
    for thetax, thetay, thetaz in itertools.product(theta, theta, theta):
        for x0, y0, z0 in itertools.product(vals, vals, vals):
            for shift in vals:
                frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                frame.xrotate(thetax)
                frame.yrotate(thetay)
                frame.zrotate(thetaz)
                frame.yshift_ref(shift)

                yversor = frame.yversor
                movement = shift * yversor
                assert all(frame.origin == np.array([x0, y0, z0]) + movement)
    
    theta = [-35, 23.5, 0]
    vals = [-123, 0, 3.5]
    for thetax, thetay, thetaz in itertools.product(theta, theta, theta):
        for x0, y0, z0 in itertools.product(vals, vals, vals):
            for shift in vals:
                frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                frame.xrotate_ref(thetax)
                frame.yrotate_ref(thetay)
                frame.zrotate_ref(thetaz)
                frame.yshift_ref(shift)

                yversor = frame.yversor
                movement = shift * yversor
                assert all(frame.origin == np.array([x0, y0, z0]) + movement)


def test_zshift_ref():
    '''Test zshift_ref method from ReferenceFrame class
    '''

    vals = [-123, 0, 0.5, 23, 56.91]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for shift in vals:
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            frame.zshift_ref(shift)
            assert all(frame.origin == np.array([x0, y0, z0 + shift]))
            assert all(frame.xversor == np.array([1, 0, 0]))
            assert all(frame.yversor == np.array([0, 1, 0]))
            assert all(frame.zversor == np.array([0, 0, 1]))

    theta = [-35, 23.5, 0]
    vals = [-123, 0, 3.5]
    for thetax, thetay, thetaz in itertools.product(theta, theta, theta):
        for x0, y0, z0 in itertools.product(vals, vals, vals):
            for shift in vals:
                frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                frame.xrotate(thetax)
                frame.yrotate(thetay)
                frame.zrotate(thetaz)
                frame.zshift_ref(shift)

                zversor = frame.zversor
                movement = shift * zversor
                assert all(frame.origin == np.array([x0, y0, z0]) + movement)
    
    theta = [-35, 23.5, 0]
    vals = [-123, 0, 3.5]
    for thetax, thetay, thetaz in itertools.product(theta, theta, theta):
        for x0, y0, z0 in itertools.product(vals, vals, vals):
            for shift in vals:
                frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                frame.xrotate_ref(thetax)
                frame.yrotate_ref(thetay)
                frame.zrotate_ref(thetaz)
                frame.zshift_ref(shift)

                zversor = frame.zversor
                movement = shift * zversor
                assert all(frame.origin == np.array([x0, y0, z0]) + movement)


def vector_rotation(vector, direction, theta):
    # unit vector in the rotation direction
    dversor = direction / np.dot(direction, direction) ** (1/2)
    # vector projection on the rotation direction
    proj = np.dot(vector, dversor)
    proj_vec = proj * dversor
    # perpendicular vector from direction to vector
    rvec = vector - proj_vec
    
    # vector aligned with direction
    if np.dot(rvec, rvec) == 0:
        return vector

    # versor
    rversor = rvec / np.dot(rvec, rvec)**(1/2)
    # 3 versor
    tversor = np.cross(rversor, dversor)
    tversor = tversor / np.dot(tversor, tversor)**(1/2)
    
    w = np.cos(np.radians(theta)) * rversor - np.sin(np.radians(theta)) * tversor
    w = w / np.dot(w, w)**(1/2)
    w = np.dot(rvec, rvec)**(1/2) * w
    
    return proj_vec + w


def test_rotate_along():
    '''Test rotate_along method from ReferenceFrame class
    '''

    # known rotations arround x
    vals = [-123, 0, 3.5]
    direction = [1, 0, 0]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along(direction, 90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert all(frame.yversor == np.array([0, 0, 1]))
        assert all(frame.zversor == np.array([0, -1, 0]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along(direction, -90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert all(frame.yversor == np.array([0, 0, -1]))
        assert all(frame.zversor == np.array([0, 1, 0]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along(direction, 45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)])
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, -1/np.sqrt(2), 1/np.sqrt(2)])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along(direction, -45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, 1/np.sqrt(2), -1/np.sqrt(2)])
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along(direction, 180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, -1, 0])
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, 0, -1])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along(direction, -180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, -1, 0])
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, 0, -1])

    # general rotation arround x
    vals = [-123, 0, 3.5]
    direction = [1, 0, 0]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for theta in np.linspace(-180, 180, 18*4+1):
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            frame.rotate_along(direction, theta)
            assert all(frame.origin == np.array([x0, y0, z0]))
            assert all(frame.xversor == np.array([1, 0, 0]))
            assert pytest.approx(frame.yversor, 0.00001) == np.array([0, np.cos(np.radians(theta)), np.sin(np.radians(theta))])
            assert pytest.approx(frame.zversor, 0.00001) == np.array([0, -np.sin(np.radians(theta)), np.cos(np.radians(theta))])
    
    # known rotations arround y
    vals = [-123, 0, 3.5]
    direction = [0, 1, 0]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along(direction, 90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([0, 0, -1]))
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert all(frame.zversor == np.array([1, 0, 0]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along(direction, -90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([0, 0, 1]))
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert all(frame.zversor == np.array([-1, 0, 0]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along(direction, 45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([1/np.sqrt(2), 0 , -1/np.sqrt(2)])
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert pytest.approx(frame.zversor, 0.00001) == np.array([1/np.sqrt(2), 0, 1/np.sqrt(2)])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along(direction, -45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([1/np.sqrt(2), 0 , 1/np.sqrt(2)])
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert pytest.approx(frame.zversor, 0.00001) == np.array([-1/np.sqrt(2), 0, 1/np.sqrt(2)])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along(direction, 180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([-1, 0, 0])
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, 0, -1])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along(direction, -180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([-1, 0, 0])
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, 0, -1])
    
    # general rotation arround y
    vals = [-123, 0, 3.5]
    direction = [0, 1, 0]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for theta in np.linspace(-180, 180, 18*4+1):
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            frame.rotate_along(direction, theta)
            assert all(frame.origin == np.array([x0, y0, z0]))
            assert pytest.approx(frame.xversor, 0.00001) == np.array([np.cos(np.radians(theta)), 0, -np.sin(np.radians(theta))])
            assert all(frame.yversor == np.array([0, 1, 0]))
            assert pytest.approx(frame.zversor, 0.00001) == np.array([np.sin(np.radians(theta)), 0, np.cos(np.radians(theta))])
    
    # known rotations arround z
    vals = [-123, 0, 3.5]
    direction = [0, 0, 1]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along(direction, 90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([0, 1, 0]))
        assert all(frame.yversor == np.array([-1, 0, 0]))
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along(direction, -90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([0, -1, 0]))
        assert all(frame.yversor == np.array([1, 0, 0]))
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along(direction, 45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([1/np.sqrt(2), 1/np.sqrt(2) , 0])
        assert pytest.approx(frame.yversor, 0.00001) == np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0])
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along(direction, -45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([1/np.sqrt(2), -1/np.sqrt(2) , 0])
        assert pytest.approx(frame.yversor, 0.00001) == np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along(direction, 180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([-1, 0 , 0])
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, -1 , 0])
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along(direction, -180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([-1, 0 , 0])
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, -1 , 0])
        assert all(frame.zversor == np.array([0, 0, 1]))

    # general rotation arround z
    vals = [-123, 0, 3.5]
    direction = [0, 0, 1]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for theta in np.linspace(-180, 180, 18*4+1):
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            frame.rotate_along(direction, theta)
            assert all(frame.origin == np.array([x0, y0, z0]))
            assert pytest.approx(frame.xversor, 0.00001) == np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta)) , 0])
            assert pytest.approx(frame.yversor, 0.00001) == np.array([-np.sin(np.radians(theta)), np.cos(np.radians(theta)) , 0])
            assert all(frame.zversor == np.array([0, 0, 1]))

    # general rotation
    theta = np.linspace(-180, 180, 18+1)
    vals = [-123, 0, 3.5]
    for lat, lon in itertools.product(theta, theta):                              
        for x0, y0, z0 in itertools.product(vals, vals, vals):
            for rot in theta:
                direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                                      np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                                      np.sin(np.radians(lat))])
                
                frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                frame.rotate_along(direction, rot)
                xversor_rot = vector_rotation(np.array([1, 0, 0]), direction, rot)
                yversor_rot = vector_rotation(np.array([0, 1, 0]), direction, rot)
                zversor_rot = vector_rotation(np.array([0, 0, 1]), direction, rot)
                assert pytest.approx(frame.xversor, 0.00001) == xversor_rot
                assert pytest.approx(frame.yversor, 0.00001) == yversor_rot
                assert pytest.approx(frame.zversor, 0.00001) == zversor_rot


def test_xrotate():
    '''Test xrotate method from ReferenceFrame class
    '''
               
    # known rotations arround x
    vals = [-123, 0, 0.5, 1, 101235]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.xrotate(90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert all(frame.yversor == np.array([0, 0, 1]))
        assert all(frame.zversor == np.array([0, -1, 0]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.xrotate(-90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert all(frame.yversor == np.array([0, 0, -1]))
        assert all(frame.zversor == np.array([0, 1, 0]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.xrotate(45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)])
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, -1/np.sqrt(2), 1/np.sqrt(2)])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.xrotate(-45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, 1/np.sqrt(2), -1/np.sqrt(2)])
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.xrotate(180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, -1, 0])
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, 0, -1])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.xrotate(-180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, -1, 0])
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, 0, -1])

    # general rotation arround x
    vals = [-123, 0, 0.5, 1, 101235]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for theta in np.linspace(-180, 180, 18*4+1):
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            frame.xrotate(theta)
            assert all(frame.origin == np.array([x0, y0, z0]))
            assert all(frame.xversor == np.array([1, 0, 0]))
            assert pytest.approx(frame.yversor, 0.00001) == np.array([0, np.cos(np.radians(theta)), np.sin(np.radians(theta))])
            assert pytest.approx(frame.zversor, 0.00001) == np.array([0, -np.sin(np.radians(theta)), np.cos(np.radians(theta))])


def test_yrotate():
    '''Test yrotate method from ReferenceFrame class
    '''

    # known rotations arround y
    vals = [-123, 0, 3.5]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.yrotate(90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([0, 0, -1]))
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert all(frame.zversor == np.array([1, 0, 0]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.yrotate(-90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([0, 0, 1]))
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert all(frame.zversor == np.array([-1, 0, 0]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.yrotate(45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([1/np.sqrt(2), 0 , -1/np.sqrt(2)])
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert pytest.approx(frame.zversor, 0.00001) == np.array([1/np.sqrt(2), 0, 1/np.sqrt(2)])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.yrotate(-45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([1/np.sqrt(2), 0 , 1/np.sqrt(2)])
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert pytest.approx(frame.zversor, 0.00001) == np.array([-1/np.sqrt(2), 0, 1/np.sqrt(2)])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.yrotate(180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([-1, 0, 0])
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, 0, -1])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.yrotate(-180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([-1, 0, 0])
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, 0, -1])
    
    # general rotation arround y
    vals = [-123, 0, 0.5, 1, 101235]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for theta in np.linspace(-180, 180, 18*4+1):
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            frame.yrotate(theta)
            assert all(frame.origin == np.array([x0, y0, z0]))
            assert pytest.approx(frame.xversor, 0.00001) == np.array([np.cos(np.radians(theta)), 0, -np.sin(np.radians(theta))])
            assert all(frame.yversor == np.array([0, 1, 0]))
            assert pytest.approx(frame.zversor, 0.00001) == np.array([np.sin(np.radians(theta)), 0, np.cos(np.radians(theta))])


def test_zrotate():
    '''Test zrotate method from ReferenceFrame class
    '''

    # known rotations arround z
    vals = [-123, 0, 3.5]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.zrotate(90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([0, 1, 0]))
        assert all(frame.yversor == np.array([-1, 0, 0]))
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.zrotate(-90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([0, -1, 0]))
        assert all(frame.yversor == np.array([1, 0, 0]))
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.zrotate(45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([1/np.sqrt(2), 1/np.sqrt(2) , 0])
        assert pytest.approx(frame.yversor, 0.00001) == np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0])
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.zrotate(-45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([1/np.sqrt(2), -1/np.sqrt(2) , 0])
        assert pytest.approx(frame.yversor, 0.00001) == np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.zrotate(180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([-1, 0 , 0])
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, -1 , 0])
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.zrotate(-180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([-1, 0 , 0])
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, -1 , 0])
        assert all(frame.zversor == np.array([0, 0, 1]))

    # general rotation arround z
    vals = [-123, 0, 0.5, 1, 101235]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for theta in np.linspace(-180, 180, 18*4+1):
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            frame.zrotate(theta)
            assert all(frame.origin == np.array([x0, y0, z0]))
            assert pytest.approx(frame.xversor, 0.00001) == np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta)) , 0])
            assert pytest.approx(frame.yversor, 0.00001) == np.array([-np.sin(np.radians(theta)), np.cos(np.radians(theta)) , 0])
            assert all(frame.zversor == np.array([0, 0, 1]))


def test_rotate_along_ref():
    '''Test rotate_along_ref method from ReferenceFrame class
    '''

    # known rotations arround x
    vals = [-123, 0, 3.5]
    direction = [1, 0, 0]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along_ref(direction, 90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert all(frame.yversor == np.array([0, 0, 1]))
        assert all(frame.zversor == np.array([0, -1, 0]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along_ref(direction, -90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert all(frame.yversor == np.array([0, 0, -1]))
        assert all(frame.zversor == np.array([0, 1, 0]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along_ref(direction, 45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)])
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, -1/np.sqrt(2), 1/np.sqrt(2)])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along_ref(direction, -45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, 1/np.sqrt(2), -1/np.sqrt(2)])
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along_ref(direction, 180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, -1, 0])
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, 0, -1])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along_ref(direction, -180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, -1, 0])
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, 0, -1])

    # general rotation arround x
    vals = [-123, 0, 3.5]
    direction = [1, 0, 0]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for theta in np.linspace(-180, 180, 18*4+1):
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            frame.rotate_along_ref(direction, theta)
            assert all(frame.origin == np.array([x0, y0, z0]))
            assert all(frame.xversor == np.array([1, 0, 0]))
            assert pytest.approx(frame.yversor, 0.00001) == np.array([0, np.cos(np.radians(theta)), np.sin(np.radians(theta))])
            assert pytest.approx(frame.zversor, 0.00001) == np.array([0, -np.sin(np.radians(theta)), np.cos(np.radians(theta))])
    
    # known rotations arround y
    vals = [-123, 0, 3.5]
    direction = [0, 1, 0]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along_ref(direction, 90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([0, 0, -1]))
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert all(frame.zversor == np.array([1, 0, 0]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along_ref(direction, -90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([0, 0, 1]))
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert all(frame.zversor == np.array([-1, 0, 0]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along_ref(direction, 45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([1/np.sqrt(2), 0 , -1/np.sqrt(2)])
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert pytest.approx(frame.zversor, 0.00001) == np.array([1/np.sqrt(2), 0, 1/np.sqrt(2)])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along_ref(direction, -45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([1/np.sqrt(2), 0 , 1/np.sqrt(2)])
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert pytest.approx(frame.zversor, 0.00001) == np.array([-1/np.sqrt(2), 0, 1/np.sqrt(2)])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along_ref(direction, 180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([-1, 0, 0])
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, 0, -1])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along_ref(direction, -180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([-1, 0, 0])
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, 0, -1])
    
    # general rotation arround y
    vals = [-123, 0, 3.5]
    direction = [0, 1, 0]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for theta in np.linspace(-180, 180, 18*4+1):
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            frame.rotate_along_ref(direction, theta)
            assert all(frame.origin == np.array([x0, y0, z0]))
            assert pytest.approx(frame.xversor, 0.00001) == np.array([np.cos(np.radians(theta)), 0, -np.sin(np.radians(theta))])
            assert all(frame.yversor == np.array([0, 1, 0]))
            assert pytest.approx(frame.zversor, 0.00001) == np.array([np.sin(np.radians(theta)), 0, np.cos(np.radians(theta))])
    
    # known rotations arround z
    vals = [-123, 0, 3.5]
    direction = [0, 0, 1]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along_ref(direction, 90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([0, 1, 0]))
        assert all(frame.yversor == np.array([-1, 0, 0]))
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along_ref(direction, -90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([0, -1, 0]))
        assert all(frame.yversor == np.array([1, 0, 0]))
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along_ref(direction, 45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([1/np.sqrt(2), 1/np.sqrt(2) , 0])
        assert pytest.approx(frame.yversor, 0.00001) == np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0])
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along_ref(direction, -45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([1/np.sqrt(2), -1/np.sqrt(2) , 0])
        assert pytest.approx(frame.yversor, 0.00001) == np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along_ref(direction, 180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([-1, 0 , 0])
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, -1 , 0])
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.rotate_along_ref(direction, -180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([-1, 0 , 0])
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, -1 , 0])
        assert all(frame.zversor == np.array([0, 0, 1]))

    # general rotation arround z
    vals = [-123, 0, 3.5]
    direction = [0, 0, 1]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for theta in np.linspace(-180, 180, 18*4+1):
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            frame.rotate_along_ref(direction, theta)
            assert all(frame.origin == np.array([x0, y0, z0]))
            assert pytest.approx(frame.xversor, 0.00001) == np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta)) , 0])
            assert pytest.approx(frame.yversor, 0.00001) == np.array([-np.sin(np.radians(theta)), np.cos(np.radians(theta)) , 0])
            assert all(frame.zversor == np.array([0, 0, 1]))

    # general rotation
    theta = np.linspace(-180, 180, 18+1)
    vals = [-123, 0, 3.5]
    for lat, lon in itertools.product(theta, theta):                              
        for x0, y0, z0 in itertools.product(vals, vals, vals):
            for rot in theta:
                direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                                      np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                                      np.sin(np.radians(lat))])
                
                frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                frame.rotate_along_ref(direction, rot)
                xversor_rot = vector_rotation(np.array([1, 0, 0]), direction, rot)
                yversor_rot = vector_rotation(np.array([0, 1, 0]), direction, rot)
                zversor_rot = vector_rotation(np.array([0, 0, 1]), direction, rot)
                assert pytest.approx(frame.xversor, 0.00001) == xversor_rot
                assert pytest.approx(frame.yversor, 0.00001) == yversor_rot
                assert pytest.approx(frame.zversor, 0.00001) == zversor_rot


def test_rotate_along_ref_rotated():

    # general rotation
    theta = np.linspace(-180, 180, 18+1)
    vals = [-123, 0, 3.5]
    for lat, lon in itertools.product(theta, theta):                              
        for x0, y0, z0 in itertools.product(vals, vals, vals):
            for rot in theta:
                direction = np.array([np.cos(np.radians(lon)) * np.cos(np.radians(lat)),
                                      np.sin(np.radians(lon)) * np.cos(np.radians(lat)),
                                      np.sin(np.radians(lat))])
                
                frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
                
                # first previous rotation
                frame.rotate_along(np.random.random(3), 360 * np.random.random() - 180)

                xversor = frame.xversor
                yversor = frame.yversor
                zversor = frame.zversor
                # direction vector in the origin system
                direction_or = direction[0] * xversor \
                               + direction[1] * yversor \
                               + direction[2] * zversor

                # intended rotation
                frame.rotate_along_ref(direction, rot)

                xversor_rot = vector_rotation(xversor, direction_or, rot)
                yversor_rot = vector_rotation(yversor, direction_or, rot)
                zversor_rot = vector_rotation(zversor, direction_or, rot)

                assert all(frame.origin == np.array([x0, y0, z0]))
                assert pytest.approx(frame.xversor, 0.00001) == xversor_rot
                assert pytest.approx(frame.yversor, 0.00001) == yversor_rot
                assert pytest.approx(frame.zversor, 0.00001) == zversor_rot


def test_xrotate_ref():
    '''Test xrotate_ref method from ReferenceFrame class
    '''

    # known rotations arround x
    vals = [-123, 0, 3.5]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.xrotate_ref(90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert all(frame.yversor == np.array([0, 0, 1]))
        assert all(frame.zversor == np.array([0, -1, 0]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.xrotate_ref(-90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert all(frame.yversor == np.array([0, 0, -1]))
        assert all(frame.zversor == np.array([0, 1, 0]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.xrotate_ref(45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)])
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, -1/np.sqrt(2), 1/np.sqrt(2)])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.xrotate_ref(-45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, 1/np.sqrt(2), -1/np.sqrt(2)])
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.xrotate_ref(180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, -1, 0])
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, 0, -1])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.xrotate_ref(-180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([1, 0, 0]))
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, -1, 0])
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, 0, -1])

    # general rotation arround x
    vals = [-123, 0, 3.5]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for theta in np.linspace(-180, 180, 18*4+1):
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            frame.xrotate_ref(theta)
            assert all(frame.origin == np.array([x0, y0, z0]))
            assert all(frame.xversor == np.array([1, 0, 0]))
            assert pytest.approx(frame.yversor, 0.00001) == np.array([0, np.cos(np.radians(theta)), np.sin(np.radians(theta))])
            assert pytest.approx(frame.zversor, 0.00001) == np.array([0, -np.sin(np.radians(theta)), np.cos(np.radians(theta))])
    
    # general rotation
    theta = np.linspace(-180, 180, 18+1)
    vals = [-123, 0, 3.5]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for rot in theta:            
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            
            # first previous rotation
            frame.rotate_along(np.random.random(3), 360 * np.random.random() - 180)

            xversor = frame.xversor
            yversor = frame.yversor
            zversor = frame.zversor
            # direction vector in the origin system
            direction_or = xversor

            # intended rotation
            frame.xrotate_ref(rot)

            xversor_rot = vector_rotation(xversor, direction_or, rot)
            yversor_rot = vector_rotation(yversor, direction_or, rot)
            zversor_rot = vector_rotation(zversor, direction_or, rot)

            assert all(frame.origin == np.array([x0, y0, z0]))
            assert pytest.approx(frame.xversor, 0.00001) == xversor_rot
            assert pytest.approx(frame.yversor, 0.00001) == yversor_rot
            assert pytest.approx(frame.zversor, 0.00001) == zversor_rot


def test_yrotate_ref():
    '''Test yrotate_ref method from ReferenceFrame class
    '''

    # known rotations arround y
    vals = [-123, 0, 3.5]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.yrotate_ref(90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([0, 0, -1]))
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert all(frame.zversor == np.array([1, 0, 0]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.yrotate_ref(-90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([0, 0, 1]))
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert all(frame.zversor == np.array([-1, 0, 0]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.yrotate_ref(45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([1/np.sqrt(2), 0 , -1/np.sqrt(2)])
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert pytest.approx(frame.zversor, 0.00001) == np.array([1/np.sqrt(2), 0, 1/np.sqrt(2)])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.yrotate_ref(-45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([1/np.sqrt(2), 0 , 1/np.sqrt(2)])
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert pytest.approx(frame.zversor, 0.00001) == np.array([-1/np.sqrt(2), 0, 1/np.sqrt(2)])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.yrotate_ref(180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([-1, 0, 0])
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, 0, -1])

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.yrotate_ref(-180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([-1, 0, 0])
        assert all(frame.yversor == np.array([0, 1, 0]))
        assert pytest.approx(frame.zversor, 0.00001) == np.array([0, 0, -1])
    
    # general rotation arround y
    vals = [-123, 0, 3.5]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for theta in np.linspace(-180, 180, 18*4+1):
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            frame.yrotate_ref(theta)
            assert all(frame.origin == np.array([x0, y0, z0]))
            assert pytest.approx(frame.xversor, 0.00001) == np.array([np.cos(np.radians(theta)), 0, -np.sin(np.radians(theta))])
            assert all(frame.yversor == np.array([0, 1, 0]))
            assert pytest.approx(frame.zversor, 0.00001) == np.array([np.sin(np.radians(theta)), 0, np.cos(np.radians(theta))])
    
    # general rotation
    theta = np.linspace(-180, 180, 18+1)
    vals = [-123, 0, 3.5]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for rot in theta:            
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            
            # first previous rotation
            frame.rotate_along(np.random.random(3), 360 * np.random.random() - 180)

            xversor = frame.xversor
            yversor = frame.yversor
            zversor = frame.zversor
            # direction vector in the origin system
            direction_or = yversor

            # intended rotation
            frame.yrotate_ref(rot)

            xversor_rot = vector_rotation(xversor, direction_or, rot)
            yversor_rot = vector_rotation(yversor, direction_or, rot)
            zversor_rot = vector_rotation(zversor, direction_or, rot)

            assert all(frame.origin == np.array([x0, y0, z0]))
            assert pytest.approx(frame.xversor, 0.00001) == xversor_rot
            assert pytest.approx(frame.yversor, 0.00001) == yversor_rot
            assert pytest.approx(frame.zversor, 0.00001) == zversor_rot


def test_zrotate_ref():
    '''Test zrotate_ref method from ReferenceFrame class
    '''

    # known rotations arround z
    vals = [-123, 0, 3.5]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.zrotate_ref(90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([0, 1, 0]))
        assert all(frame.yversor == np.array([-1, 0, 0]))
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.zrotate_ref(-90)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert all(frame.xversor == np.array([0, -1, 0]))
        assert all(frame.yversor == np.array([1, 0, 0]))
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.zrotate_ref(45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([1/np.sqrt(2), 1/np.sqrt(2) , 0])
        assert pytest.approx(frame.yversor, 0.00001) == np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0])
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.zrotate_ref(-45)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([1/np.sqrt(2), -1/np.sqrt(2) , 0])
        assert pytest.approx(frame.yversor, 0.00001) == np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.zrotate_ref(180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([-1, 0 , 0])
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, -1 , 0])
        assert all(frame.zversor == np.array([0, 0, 1]))

        frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
        frame.zrotate_ref(-180)
        assert all(frame.origin == np.array([x0, y0, z0]))
        assert pytest.approx(frame.xversor, 0.00001) == np.array([-1, 0 , 0])
        assert pytest.approx(frame.yversor, 0.00001) == np.array([0, -1 , 0])
        assert all(frame.zversor == np.array([0, 0, 1]))

    # general rotation arround z
    vals = [-123, 0, 3.5]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for theta in np.linspace(-180, 180, 18*4+1):
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            frame.zrotate_ref(theta)
            assert all(frame.origin == np.array([x0, y0, z0]))
            assert pytest.approx(frame.xversor, 0.00001) == np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta)) , 0])
            assert pytest.approx(frame.yversor, 0.00001) == np.array([-np.sin(np.radians(theta)), np.cos(np.radians(theta)) , 0])
            assert all(frame.zversor == np.array([0, 0, 1]))
    
    # general rotation
    theta = np.linspace(-180, 180, 18+1)
    vals = [-123, 0, 3.5]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for rot in theta:            
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)
            
            # first previous rotation
            frame.rotate_along(np.random.random(3), 360 * np.random.random() - 180)

            xversor = frame.xversor
            yversor = frame.yversor
            zversor = frame.zversor
            # direction vector in the origin system
            direction_or = zversor

            # intended rotation
            frame.zrotate_ref(rot)

            xversor_rot = vector_rotation(xversor, direction_or, rot)
            yversor_rot = vector_rotation(yversor, direction_or, rot)
            zversor_rot = vector_rotation(zversor, direction_or, rot)

            assert all(frame.origin == np.array([x0, y0, z0]))
            assert pytest.approx(frame.xversor, 0.00001) == xversor_rot
            assert pytest.approx(frame.yversor, 0.00001) == yversor_rot
            assert pytest.approx(frame.zversor, 0.00001) == zversor_rot


def test_move_to_origin():

    frame = ReferenceFrame()
    frame.move_to_origin()
    assert all(frame.origin == np.array([0, 0, 0]))
    assert all(frame.xversor == np.array([1, 0, 0]))
    assert all(frame.yversor == np.array([0, 1, 0]))
    assert all(frame.zversor == np.array([0, 0, 1]))

    vals = [-123, 0, 0.5, 1, 101235]
    for xshift, yshift, zshift in itertools.product(vals, vals, vals):
        frame = ReferenceFrame()
        frame.shift(xshift, yshift, zshift)
        frame.move_to_origin()
        assert all(frame.origin == np.array([0, 0, 0]))
        assert pytest.approx(frame.xversor, 0.000001) == np.array([1, 0, 0])
        assert pytest.approx(frame.yversor, 0.000001) == np.array([0, 1, 0])
        assert pytest.approx(frame.zversor, 0.000001) == np.array([0, 0, 1])

        frame = ReferenceFrame()
        frame.rotate_along(np.random.random(3), 360 * np.random.random() - 180)
        frame.shift_ref(xshift, yshift, zshift)
        frame.move_to_origin()
        assert all(frame.origin == np.array([0, 0, 0]))
        assert pytest.approx(frame.xversor, 0.000001) == np.array([1, 0, 0])
        assert pytest.approx(frame.yversor, 0.000001) == np.array([0, 1, 0])
        assert pytest.approx(frame.zversor, 0.000001) == np.array([0, 0, 1])


def test_o2r():
    '''Test o2r method from ReferenceFrame class
    '''

    # general rotation
    theta = np.linspace(-180, 180, 18+1)
    vals = [-123, 0, 3.5]
    coords = [-23.5, -1, 0, 0.7, 1239.4]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for rot in theta:
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)        
            # first previous rotation
            frame.rotate_along(np.random.random(3), rot)

            xversor = frame.xversor
            yversor = frame.yversor
            zversor = frame.zversor
            for xcoord, ycoord, zcoord in itertools.product(coords, coords, coords):
                vector = np.array([xcoord, ycoord, zcoord])
                vector_r = frame.o2r(vector)
                xcoord_r = np.dot(xversor, vector)
                ycoord_r = np.dot(yversor, vector)
                zcoord_r = np.dot(zversor, vector)
                assert pytest.approx(vector_r[0], 0.00001) == xcoord_r
                assert pytest.approx(vector_r[1], 0.00001) == ycoord_r
                assert pytest.approx(vector_r[2], 0.00001) == zcoord_r
    

def test_r2o():
    '''Test r2o method from ReferenceFrame class
    '''

    # general rotation
    theta = np.linspace(-180, 180, 18+1)
    vals = [-123, 0, 3.5]
    coords = [-23.5, -1, 0, 0.7, 1239.4]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for rot in theta:
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)        
            # first previous rotation
            frame.rotate_along(np.random.random(3), rot)

            xversor = frame.xversor
            yversor = frame.yversor
            zversor = frame.zversor
            for xcoord, ycoord, zcoord in itertools.product(coords, coords, coords):
                vector = np.array([xcoord, ycoord, zcoord])
                vector_o = frame.r2o(vector)
                expected = vector[0] * xversor + vector[1] * yversor + vector[2] * zversor
                assert pytest.approx(vector_o, 0.00001) == expected


def test_pos_o2r():
    '''Test pos_o2r method from ReferenceFrame class
    '''
    
    # general rotation
    theta = np.linspace(-180, 180, 18+1)
    vals = [-123, 0, 3.5]
    coords = [-23.5, -1, 0, 0.7, 1239.4]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for rot in theta:
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)        
            # first previous rotation
            frame.rotate_along(np.random.random(3), rot)

            for xcoord, ycoord, zcoord in itertools.product(coords, coords, coords):
                pos_o = np.array([xcoord, ycoord, zcoord])
                pos_r = frame.pos_o2r(pos_o)
                expected = frame.o2r(pos_o - frame.origin)
                assert pytest.approx(pos_r, 0.00001) == expected


def test_pos_r2o():
    '''Test pos_r2o method from ReferenceFrame class
    '''

    # general rotation
    theta = np.linspace(-180, 180, 18+1)
    vals = [-123, 0, 3.5]
    coords = [-23.5, -1, 0, 0.7, 1239.4]
    for x0, y0, z0 in itertools.product(vals, vals, vals):
        for rot in theta:
            frame = ReferenceFrame(xcoord=x0, ycoord=y0, zcoord=z0)        
            # first previous rotation
            frame.rotate_along(np.random.random(3), rot)

            for xcoord, ycoord, zcoord in itertools.product(coords, coords, coords):
                pos_r = np.array([xcoord, ycoord, zcoord])
                pos_o = frame.pos_r2o(pos_r)
                expected = frame.origin + pos_r[0] * frame.xversor \
                           + pos_r[1] * frame.yversor + pos_r[2] * frame.zversor
                assert pytest.approx(pos_o, 0.00001) == expected