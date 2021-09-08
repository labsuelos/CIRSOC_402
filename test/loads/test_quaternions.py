'''Module for testing the Quaternion class in
cirsoc_402.loads.quaternion
'''
import numpy as np
import pytest
import itertools

from cirsoc_402.load import Quaternion

def test_init():
    '''Test __init__ method from Quaternion class
    '''
    xvec = np.logspace(-3, 3, 10)
    xvec = np.concatenate([-xvec, [0], xvec])

    for q0, q1, q2, q3 in itertools.product(xvec, xvec, xvec, xvec):
        q = Quaternion(q0, q1, q2, q3)
        assert q._coord0 == q0
        assert q._coord1 == q1
        assert q._coord2 == q2
        assert q._coord3 == q3
    
    # test input nan
    q = Quaternion(np.nan, np.nan, np.nan, np.nan)
    assert np.isnan(q._coord0) == True
    assert np.isnan(q._coord1) == True
    assert np.isnan(q._coord2) == True
    assert np.isnan(q._coord3) == True

    q = Quaternion(np.nan, 1, 2, 3)
    assert np.isnan(q._coord0) == True
    assert q._coord1 == 1
    assert q._coord2 == 2
    assert q._coord3 == 3

    q = Quaternion(np.nan, 1, np.nan, 3)
    assert np.isnan(q._coord0) == True
    assert q._coord1 == 1
    assert np.isnan(q._coord2) == True
    assert q._coord3 == 3

    # Check type error when creating
    with pytest.raises(TypeError):
        q = Quaternion('x', 2, 3, 4)
    with pytest.raises(TypeError):
        q = Quaternion(1, '2', 3, 4)
    with pytest.raises(TypeError):
        q = Quaternion(1, 2, '3', 4)
    with pytest.raises(TypeError):
        q = Quaternion(1, 2, 3, '4')
    with pytest.raises(TypeError):
        q = Quaternion('1', '2', '3' '4')
    with pytest.raises(TypeError):
        q = Quaternion(1, '2', 3, '4')
    with pytest.raises(TypeError):
        q = Quaternion([1], 2, 3, 4)
    with pytest.raises(TypeError):
        q = Quaternion(1, [2], 3, 4)
    with pytest.raises(TypeError):
        q = Quaternion(1, 2, [3], 4)
    with pytest.raises(TypeError):
        q = Quaternion(1, 2, 3, [4])
    with pytest.raises(TypeError):
        q = Quaternion([1], [2], [3], [4])
    with pytest.raises(TypeError):
        q = Quaternion(1, [2], 3, [4])
    

def test_repr():
    '''Test __repr__ method from Quaternion class
    '''

    q = Quaternion(1, 2, 3, 4)
    assert q.__repr__() == '(1.00, 2.00, 3.00, 4.00)'

    q = Quaternion(1.00, 2.0, 3, 4)
    assert q.__repr__() == '(1.00, 2.00, 3.00, 4.00)'

    q = Quaternion(1.001, 2.005, 3, 4)
    assert q.__repr__() == '(1.00, 2.00, 3.00, 4.00)'

    q = Quaternion(1, np.nan, np.nan, 4)
    assert q.__repr__() == '(1.00, nan, nan, 4.00)'


def test_add():
    '''Test __add__ method from Quaternion class
    '''

    vals = [-123, 0, 0.5, 1, 101235]
    for q0, q1, q2, q3 in itertools.product(vals, vals, vals, vals):
        q1 = Quaternion(q0, q1, q2, q3)
        for r0, r1, r2, r3 in itertools.product(vals, vals, vals, vals):
            q2 = Quaternion(r0, r1, r2, r3)
            q3 = q1 + q2
            assert q3._coord0 == q1._coord0 + q2._coord0
            assert q3._coord1 == q1._coord1 + q2._coord1
            assert q3._coord2 == q1._coord2 + q2._coord2
            assert q3._coord3 == q1._coord3 + q2._coord3
            assert q3._coord0 == q2._coord0 + q1._coord0
            assert q3._coord1 == q2._coord1 + q1._coord1
            assert q3._coord2 == q2._coord2 + q1._coord2
            assert q3._coord3 == q2._coord3 + q1._coord3

            q3 = q2 + q1
            assert q3._coord0 == q1._coord0 + q2._coord0
            assert q3._coord1 == q1._coord1 + q2._coord1
            assert q3._coord2 == q1._coord2 + q2._coord2
            assert q3._coord3 == q1._coord3 + q2._coord3
            assert q3._coord0 == q2._coord0 + q1._coord0
            assert q3._coord1 == q2._coord1 + q1._coord1
            assert q3._coord2 == q2._coord2 + q1._coord2
            assert q3._coord3 == q2._coord3 + q1._coord3
    
    q1 = Quaternion(1, 2, 3, np.nan)
    q2 = Quaternion(1, np.nan, 3, 5)
    q3 = q1 + q2
    assert q3._coord0 == q1._coord0 + q2._coord0
    assert np.isnan(q3._coord1) == True
    assert q3._coord2 == q1._coord2 + q2._coord2
    assert np.isnan(q3._coord3) == True

    q3 = q2 + q1
    assert q3._coord0 == q1._coord0 + q2._coord0
    assert np.isnan(q3._coord1) == True
    assert q3._coord2 == q1._coord2 + q2._coord2
    assert np.isnan(q3._coord3) == True


    # Check type error
    q = Quaternion(1, 2, 3, 4)
    with pytest.raises(TypeError):
        _ = q + 'a'
    with pytest.raises(TypeError):
        _ = q + [1, 2, 3, 4]
    with pytest.raises(TypeError):
        _ = q + np.array([1, 2, 3, 4])


def test_sub():
    '''Test __sub__ method from Quaternion class
    '''

    vals = [-123, 0, 0.5, 1, 101235]
    for q0, q1, q2, q3 in itertools.product(vals, vals, vals, vals):
        q1 = Quaternion(q0, q1, q2, q3)
        for r0, r1, r2, r3 in itertools.product(vals, vals, vals, vals):
            q2 = Quaternion(r0, r1, r2, r3)
            q3 = q1 - q2
            assert q3._coord0 == q1._coord0 - q2._coord0
            assert q3._coord1 == q1._coord1 - q2._coord1
            assert q3._coord2 == q1._coord2 - q2._coord2
            assert q3._coord3 == q1._coord3 - q2._coord3

            q3 = q2 - q1
            assert q3._coord0 == q2._coord0 - q1._coord0
            assert q3._coord1 == q2._coord1 - q1._coord1
            assert q3._coord2 == q2._coord2 - q1._coord2
            assert q3._coord3 == q2._coord3 - q1._coord3
    
    q1 = Quaternion(1, 2, 3, np.nan)
    q2 = Quaternion(1, np.nan, 3, 5)
    q3 = q1 - q2
    assert q3._coord0 == q1._coord0 - q2._coord0
    assert np.isnan(q3._coord1) == True
    assert q3._coord2 == q1._coord2 - q2._coord2
    assert np.isnan(q3._coord3) == True

    q3 = q2 - q1
    assert q3._coord0 == q2._coord0 - q1._coord0
    assert np.isnan(q3._coord1) == True
    assert q3._coord2 == q2._coord2 - q1._coord2
    assert np.isnan(q3._coord3) == True

    # Check type error
    q = Quaternion(1, 2, 3, 4)
    with pytest.raises(TypeError):
        _ = q - 'a'
    with pytest.raises(TypeError):
        _ = q - [1, 2, 3, 4]
    with pytest.raises(TypeError):
        _ = q - np.array([1, 2, 3, 4])


def test_scalar_mul():
    '''Test __mul__ and __rmul__ methods from Quaternion class when
    multiplying by a scalar
    '''

    # Test multiplication by scalar
    q = Quaternion(1, 2, 3, 4)
    for scl in [-123, -45, -3, -0.4, 0, 0.65, 4, 123]:
        q2 = q  *scl
        assert q2._coord0 == q._coord0 * scl
        assert q2._coord1 == q._coord1 * scl
        assert q2._coord2 == q._coord2 * scl
        assert q2._coord3 == q._coord3 * scl
        assert q2._coord0 == scl * q._coord0
        assert q2._coord1 == scl * q._coord1
        assert q2._coord2 == scl * q._coord2
        assert q2._coord3 == scl * q._coord3

        q2 = scl * q
        assert q2._coord0 == q._coord0 * scl
        assert q2._coord1 == q._coord1 * scl
        assert q2._coord2 == q._coord2 * scl
        assert q2._coord3 == q._coord3 * scl
        assert q2._coord0 == scl * q._coord0
        assert q2._coord1 == scl * q._coord1
        assert q2._coord2 == scl * q._coord2
        assert q2._coord3 == scl * q._coord3
    
    # Test nan multiplication
    q = Quaternion(1, 2, 3, 4)
    q2 = q * np.nan
    assert np.isnan(q2._coord0) == True
    assert np.isnan(q2._coord1) == True
    assert np.isnan(q2._coord2) == True
    assert np.isnan(q2._coord3) == True

    q2 = np.nan * q
    assert np.isnan(q2._coord0) == True
    assert np.isnan(q2._coord1) == True
    assert np.isnan(q2._coord2) == True
    assert np.isnan(q2._coord3) == True

    q = Quaternion(1, np.nan, 3, np.nan)
    q2 = q * 2
    assert q2._coord0 == 2
    assert np.isnan(q2._coord1) == True
    assert q2._coord2 == 6
    assert np.isnan(q2._coord3) == True
    
    q2 = 2 * q
    assert q2._coord0 == 2
    assert np.isnan(q2._coord1) == True
    assert q2._coord2 == 6
    assert np.isnan(q2._coord3) == True

    # Check type error
    q = Quaternion(1, 2, 3, 4)
    with pytest.raises(TypeError):
        _ = q * 'a'
    with pytest.raises(TypeError):
        _ = 'a' * q 
    with pytest.raises(TypeError):
        _ = q * [1]
    with pytest.raises(TypeError):
        _ = [1] * q 


def test_quaternion_mul():
    '''Test __mul__ and __rmul__ methods from Quaternion class when
    multiplying by another quaternion
    '''
    vals = [-123, 0, 0.5, 1, 101235]
    for q0, q1, q2, q3 in itertools.product(vals, vals, vals, vals):
        quat1 = Quaternion(q0, q1, q2, q3)
        for r0, r1, r2, r3 in itertools.product(vals, vals, vals, vals):
            quat2 = Quaternion(r0, r1, r2, r3)
            quat3 = quat1 * quat2
            # q0p0 − q1p1 − q2p2 − q3p3
            assert quat3._coord0 == q0 * r0 - q1 * r1 - q2 * r2 - q3 * r3
            # q0p1 + q1p0 + q2p3 − q3p2
            assert quat3._coord1 == q0 * r1 + q1 * r0 + q2 * r3 - q3 * r2
            # q0p2 + q2p0 − q1p3 + q3p1
            assert quat3._coord2 == q0 * r2 + q2 * r0 - q1 * r3 + q3 * r1
            # q0p3 + q3p0 + q1p2 − q2p1
            assert quat3._coord3 == q0 * r3 + q3 * r0 + q1 * r2 - q2 * r1

            quat3 = quat2 * quat1
            # q0p0 − q1p1 − q2p2 − q3p3
            assert quat3._coord0 == r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
            # q0p1 + q1p0 + q2p3 − q3p2
            assert quat3._coord1 == r0 * q1 + r1 * q0 + r2 * q3 - r3 * q2
            # q0p2 + q2p0 − q1p3 + q3p1
            assert quat3._coord2 == r0 * q2 + r2 * q0 - r1 * q3 + r3 * q1
            # q0p3 + q3p0 + q1p2 − q2p1
            assert quat3._coord3 == r0 * q3 + r3 * q0 + r1 * q2 - r2 * q1

    q0, q1, q2, q3 = (1, 2, np.nan, 4)
    r0, r1, r2, r3 = (1, 2, 3, 4)
    quat1 = Quaternion(q0, q1, q2, q3)
    quat2 = Quaternion(r0, r1, r2, r3)
    quat3 = quat1 * quat2
    # q0p0 − q1p1 − q2p2 − q3p3
    assert np.isnan(quat3._coord0) == True
    # q0p1 + q1p0 + q2p3 − q3p2
    assert np.isnan(quat3._coord1) == True
    # q0p2 + q2p0 − q1p3 + q3p1
    assert np.isnan(quat3._coord2) == True
    # q0p3 + q3p0 + q1p2 − q2p1
    assert np.isnan(quat3._coord3) == True


    q0, q1, q2, q3 = (1, 2, np.nan, 4)
    r0, r1, r2, r3 = (1, np.nan, 3, 4)
    quat1 = Quaternion(q0, q1, q2, q3)
    quat2 = Quaternion(r0, r1, r2, r3)
    quat3 = quat1 * quat2
    # q0p0 − q1p1 − q2p2 − q3p3
    assert np.isnan(quat3._coord0) == True
    # q0p1 + q1p0 + q2p3 − q3p2
    assert np.isnan(quat3._coord1) == True
    # q0p2 + q2p0 − q1p3 + q3p1
    assert np.isnan(quat3._coord2) == True
    # q0p3 + q3p0 + q1p2 − q2p1
    assert np.isnan(quat3._coord3) == True


    q0, q1, q2, q3 = (np.nan, np.nan, np.nan, np.nan)
    r0, r1, r2, r3 = (1, 2, 3, 4)
    quat1 = Quaternion(q0, q1, q2, q3)
    quat2 = Quaternion(r0, r1, r2, r3)
    quat3 = quat1 * quat2
    # q0p0 − q1p1 − q2p2 − q3p3
    assert np.isnan(quat3._coord0) == True
    # q0p1 + q1p0 + q2p3 − q3p2
    assert np.isnan(quat3._coord1) == True
    # q0p2 + q2p0 − q1p3 + q3p1
    assert np.isnan(quat3._coord2) == True
    # q0p3 + q3p0 + q1p2 − q2p1
    assert np.isnan(quat3._coord3) == True

    # Check type error
    q = Quaternion(1, 2, 3, 4)
    with pytest.raises(TypeError):
        _ = q * [1, 2, 3, 4]
    with pytest.raises(TypeError):
        _ = [1, 2, 3, 4] * q 


def test_division():
    '''Test __truediv__ method from Quaternion class
    '''

    # Test multiplication by scalar
    q = Quaternion(1, 2, 3, 4)
    for scl in [-123, -45, -3, -0.4, 0.65, 4, 123]:
        q2 = q  / scl
        assert q2._coord0 == q._coord0 / scl
        assert q2._coord1 == q._coord1 / scl
        assert q2._coord2 == q._coord2 / scl
        assert q2._coord3 == q._coord3 / scl

    # Test nan multiplication
    q = Quaternion(1, 2, 3, 4)
    q2 = q / np.nan
    assert np.isnan(q2._coord0) == True
    assert np.isnan(q2._coord1) == True
    assert np.isnan(q2._coord2) == True
    assert np.isnan(q2._coord3) == True

    q = Quaternion(1, np.nan, 3, np.nan)
    q2 = q / 2
    assert q2._coord0 == 0.5
    assert np.isnan(q2._coord1) == True
    assert q2._coord2 == 1.5
    assert np.isnan(q2._coord3) == True

    # Check type error
    q = Quaternion(1, 2, 3, 4)
    with pytest.raises(TypeError):
        _ = q / '2'
    with pytest.raises(TypeError):
        _ = q / [1]


def test_iadd():
    '''Test __iadd__ method from Quaternion class
    '''

    vals = [-123, 0, 0.5, 1, 101235]
    for q0, q1, q2, q3 in itertools.product(vals, vals, vals, vals):
        for r0, r1, r2, r3 in itertools.product(vals, vals, vals, vals):
            quat1 = Quaternion(q0, q1, q2, q3)
            quat2 = Quaternion(r0, r1, r2, r3)
            quat1 += quat2
            assert quat1._coord0 == q0 + r0
            assert quat1._coord1 == q1 + r1
            assert quat1._coord2 == q2 + r2
            assert quat1._coord3 == q3 + r3
    
    q1 = Quaternion(1, 2, 3, np.nan)
    q2 = Quaternion(1, np.nan, 3, 5)
    q1 += q2
    assert q1._coord0 == 2
    assert np.isnan(q1._coord1) == True
    assert q1._coord2 == 6
    assert np.isnan(q1._coord3) == True


    # Check type error
    q = Quaternion(1, 2, 3, 4)
    with pytest.raises(TypeError):
        q += 'a'
    with pytest.raises(TypeError):
        q += [1, 2, 3, 4]
    with pytest.raises(TypeError):
        q += np.array([1, 2, 3, 4])


def test_isub():
    '''Test __isub__ method from Quaternion class
    '''

    vals = [-123, 0, 0.5, 1, 101235]
    for q0, q1, q2, q3 in itertools.product(vals, vals, vals, vals):
        for r0, r1, r2, r3 in itertools.product(vals, vals, vals, vals):
            quat1 = Quaternion(q0, q1, q2, q3)
            quat2 = Quaternion(r0, r1, r2, r3)
            quat1 -= quat2
            assert quat1._coord0 == q0 - r0
            assert quat1._coord1 == q1 - r1
            assert quat1._coord2 == q2 - r2
            assert quat1._coord3 == q3 - r3
    
    q1 = Quaternion(1, 2, 3, np.nan)
    q2 = Quaternion(1, np.nan, 4, 5)
    q1 -= q2
    assert q1._coord0 == 0
    assert np.isnan(q1._coord1) == True
    assert q1._coord2 == -1
    assert np.isnan(q1._coord3) == True


    # Check type error
    q = Quaternion(1, 2, 3, 4)
    with pytest.raises(TypeError):
        q -= 'a'
    with pytest.raises(TypeError):
        q -= [1, 2, 3, 4]
    with pytest.raises(TypeError):
        q -= np.array([1, 2, 3, 4])


def test_scalar_imul_():
    '''Test __imul__ method from Quaternion class when multiplying by a
    scalar
    '''

    # Test multiplication by scalar
    for scl in [-123, -45, -3, -0.4, 0, 0.65, 4, 123]:
        q = Quaternion(-1, 2, 0, 4)
        q *= scl
        assert q._coord0 == -1 * scl
        assert q._coord1 == 2 * scl
        assert q._coord2 == 0 * scl
        assert q._coord3 == 4 * scl

    # Test nan multiplication
    q2 = Quaternion(1, 2, 3, 4)
    q2 *= np.nan
    assert np.isnan(q2._coord0) == True
    assert np.isnan(q2._coord1) == True
    assert np.isnan(q2._coord2) == True
    assert np.isnan(q2._coord3) == True

    q2 = Quaternion(1, np.nan, 3, np.nan)
    q2 *= 2
    assert q2._coord0 == 2
    assert np.isnan(q2._coord1) == True
    assert q2._coord2 == 6
    assert np.isnan(q2._coord3) == True
    
    # Check type error
    q = Quaternion(1, 2, 3, 4)
    with pytest.raises(TypeError):
        q *= 'a'
    with pytest.raises(TypeError):
        q *= [1]


def test_quaternion_imul():
    '''Test __mul__ and __rmul__ methods from Quaternion class when
    multiplying by another quaternion
    '''
    vals = [-123, 0, 0.5, 1, 101235]
    for q0, q1, q2, q3 in itertools.product(vals, vals, vals, vals):
        for r0, r1, r2, r3 in itertools.product(vals, vals, vals, vals):
            quat1 = Quaternion(q0, q1, q2, q3)
            quat2 = Quaternion(r0, r1, r2, r3)
            quat1 *= quat2
            # q0p0 − q1p1 − q2p2 − q3p3
            assert quat1._coord0 == q0 * r0 - q1 * r1 - q2 * r2 - q3 * r3
            # q0p1 + q1p0 + q2p3 − q3p2
            assert quat1._coord1 == q0 * r1 + q1 * r0 + q2 * r3 - q3 * r2
            # q0p2 + q2p0 − q1p3 + q3p1
            assert quat1._coord2 == q0 * r2 + q2 * r0 - q1 * r3 + q3 * r1
            # q0p3 + q3p0 + q1p2 − q2p1
            assert quat1._coord3 == q0 * r3 + q3 * r0 + q1 * r2 - q2 * r1


    q0, q1, q2, q3 = (1, 2, np.nan, 4)
    r0, r1, r2, r3 = (1, 2, 3, 4)
    quat1 = Quaternion(q0, q1, q2, q3)
    quat2 = Quaternion(r0, r1, r2, r3)
    quat1 *= quat2
    # q0p0 − q1p1 − q2p2 − q3p3
    assert np.isnan(quat1._coord0) == True
    # q0p1 + q1p0 + q2p3 − q3p2
    assert np.isnan(quat1._coord1) == True
    # q0p2 + q2p0 − q1p3 + q3p1
    assert np.isnan(quat1._coord2) == True
    # q0p3 + q3p0 + q1p2 − q2p1
    assert np.isnan(quat1._coord3) == True


    q0, q1, q2, q3 = (1, 2, np.nan, 4)
    r0, r1, r2, r3 = (1, np.nan, 3, 4)
    quat1 = Quaternion(q0, q1, q2, q3)
    quat2 = Quaternion(r0, r1, r2, r3)
    quat1 *= quat2
    # q0p0 − q1p1 − q2p2 − q3p3
    assert np.isnan(quat1._coord0) == True
    # q0p1 + q1p0 + q2p3 − q3p2
    assert np.isnan(quat1._coord1) == True
    # q0p2 + q2p0 − q1p3 + q3p1
    assert np.isnan(quat1._coord2) == True
    # q0p3 + q3p0 + q1p2 − q2p1
    assert np.isnan(quat1._coord3) == True

    q0, q1, q2, q3 = (np.nan, np.nan, np.nan, np.nan)
    r0, r1, r2, r3 = (1, 2, 3, 4)
    quat1 = Quaternion(q0, q1, q2, q3)
    quat2 = Quaternion(r0, r1, r2, r3)
    quat1 *= quat2
    # q0p0 − q1p1 − q2p2 − q3p3
    assert np.isnan(quat1._coord0) == True
    # q0p1 + q1p0 + q2p3 − q3p2
    assert np.isnan(quat1._coord1) == True
    # q0p2 + q2p0 − q1p3 + q3p1
    assert np.isnan(quat1._coord2) == True
    # q0p3 + q3p0 + q1p2 − q2p1
    assert np.isnan(quat1._coord3) == True

    # Check type error
    q = Quaternion(1, 2, 3, 4)
    with pytest.raises(TypeError):
        q *= [1, 2, 3, 4]


def test_idivision():
    '''Test __truediv__ method from Quaternion class
    '''

    # Test multiplication by scalar
    
    for scl in [-123, -45, -3, -0.4, 0.65, 4, 123]:
        q2 = Quaternion(1, -2, 0, 4)
        q2  /= scl
        assert q2._coord0 == 1 / scl
        assert q2._coord1 == -2 / scl
        assert q2._coord2 == 0 / scl
        assert q2._coord3 == 4 / scl

    # Test nan multiplication
    q2 = Quaternion(1, 2, 3, 4)
    q2 /= np.nan
    assert np.isnan(q2._coord0) == True
    assert np.isnan(q2._coord1) == True
    assert np.isnan(q2._coord2) == True
    assert np.isnan(q2._coord3) == True

    q2 = Quaternion(1, np.nan, 3, np.nan)
    q2 /= 2
    assert q2._coord0 == 0.5
    assert np.isnan(q2._coord1) == True
    assert q2._coord2 == 1.5
    assert np.isnan(q2._coord3) == True

    # Check type error
    q2 = Quaternion(1, 2, 3, 4)
    with pytest.raises(TypeError):
        q2 /= '2'
    with pytest.raises(TypeError):
        q2 /= [1]


def test_conj():
    '''Test __conj__ method from Quaternion class
    '''
    vals = [-123, 0, 0.5, 1, 101235]
    for q0, q1, q2, q3 in itertools.product(vals, vals, vals, vals):
        q1 = Quaternion(q0, q1, q2, q3)
        q2 = q1.conj()
        assert q2._coord0 == q1._coord0
        assert q2._coord1 == -q1._coord1
        assert q2._coord2 == -q1._coord2
        assert q2._coord3 == -q1._coord3
    
    q1 = Quaternion(1, 2, 3, np.nan)
    q2 = q1.conj()
    assert q2._coord0 == q1._coord0
    assert q2._coord1 == -q1._coord1
    assert q2._coord2 == -q1._coord2
    assert np.isnan(q2._coord3) == True

    q1 = Quaternion(1, np.nan, 3, np.nan)
    q2 = q1.conj()
    assert q2._coord0 == q1._coord0
    assert np.isnan(q2._coord1) == True
    assert q2._coord2 == -q1._coord2
    assert np.isnan(q2._coord3) == True

    q1 = Quaternion(np.nan, np.nan, np.nan, np.nan)
    q2 = q1.conj()
    assert np.isnan(q2._coord0) == True
    assert np.isnan(q2._coord1) == True
    assert np.isnan(q2._coord2) == True
    assert np.isnan(q2._coord3) == True


def test_norm():
    '''Test __norm__ method from Quaternion class
    '''
    vals = [-123, 0, 0.5, 1, 101235]
    for q0, q1, q2, q3 in itertools.product(vals, vals, vals, vals):
        quat = Quaternion(q0, q1, q2, q3)
        norm = quat.norm()
        expected = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
        assert norm == pytest.approx(expected, 0.001)


def test_inverse():
    
    vals = [-123, 0, 0.5, 1, 101235]
    for q0, q1, q2, q3 in itertools.product(vals, vals, vals, vals):
        if q0 == 0  and q1 == 0 and q2 ==0 and q3 ==0:
            continue
        quat = Quaternion(q0, q1, q2, q3)
        inv = quat.inv()
        norm = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
        assert pytest.approx(inv._coord0, 0.01) == q0 / norm**2
        assert pytest.approx(inv._coord1, 0.01) == - q1 / norm**2
        assert pytest.approx(inv._coord2, 0.01) == - q2 / norm**2
        assert pytest.approx(inv._coord3, 0.01) == - q3 / norm**2

    quat = Quaternion(0, 0, 0, 0)
    inv = quat.inv()
    assert np.isnan(inv._coord0) == True
    assert np.isnan(inv._coord1) == True
    assert np.isnan(inv._coord2) == True
    assert np.isnan(inv._coord3) == True

    quat = Quaternion(1, 2, 3, np.nan)
    inv = quat.inv()
    assert np.isnan(inv._coord0) == True
    assert np.isnan(inv._coord1) == True
    assert np.isnan(inv._coord2) == True
    assert np.isnan(inv._coord3) == True

    quat = Quaternion(1, np.nan, 3, np.nan)
    inv = quat.inv()
    assert np.isnan(inv._coord0) == True
    assert np.isnan(inv._coord1) == True
    assert np.isnan(inv._coord2) == True
    assert np.isnan(inv._coord3) == True

    quat = Quaternion(np.nan, np.nan, np.nan, np.nan)
    inv = quat.inv()
    assert np.isnan(inv._coord0) == True
    assert np.isnan(inv._coord1) == True
    assert np.isnan(inv._coord2) == True
    assert np.isnan(inv._coord3) == True


def test_rotate():
    '''Test rotate method from Quaternion class.
    '''
    vals = [-123, 0, 0.5, 1, 101235]
    for q0, q1, q2, q3 in itertools.product(vals, vals, vals, vals):
        for r0, r1, r2, r3 in itertools.product(vals, vals, vals, vals):
            quat1 = Quaternion(q0, q1, q2, q3)
            quat2 = Quaternion(r0, r1, r2, r3)
            quat3 = quat1.rotate(quat2)

            # transpose quat1
            s0 = q0
            s1 = -q1
            s2 = -q2
            s3 = -q3

            # quat1 * quat2pytest -kro
            x0 = q0 * r0 - q1 * r1 - q2 * r2 - q3 * r3
            x1 = q0 * r1 + q1 * r0 + q2 * r3 - q3 * r2
            x2 = q0 * r2 + q2 * r0 - q1 * r3 + q3 * r1
            x3 = q0 * r3 + q3 * r0 + q1 * r2 - q2 * r1
            # q0p0 − q1p1 − q2p2 − q3p3
            assert quat3._coord0 == x0 * s0 - x1 * s1 - x2 * s2 - x3 * s3
            # q0p1 + q1p0 + q2p3 − q3p2
            assert quat3._coord1 == x0 * s1 + x1 * s0 + x2 * s3 - x3 * s2
            # q0p2 + q2p0 − q1p3 + q3p1
            assert quat3._coord2 == x0 * s2 + x2 * s0 - x1 * s3 + x3 * s1
            # q0p3 + q3p0 + q1p2 − q2p1
            assert quat3._coord3 == x0 * s3 + x3 * s0 + x1 * s2 - x2 * s1

    q0, q1, q2, q3 = (1, 2, np.nan, 4)
    r0, r1, r2, r3 = (1, 2, 3, 4)
    quat1 = Quaternion(q0, q1, q2, q3)
    quat2 = Quaternion(r0, r1, r2, r3)
    quat3 = quat1.rotate(quat2)
    # q0p0 − q1p1 − q2p2 − q3p3
    assert np.isnan(quat3._coord0) == True
    # q0p1 + q1p0 + q2p3 − q3p2
    assert np.isnan(quat3._coord1) == True
    # q0p2 + q2p0 − q1p3 + q3p1
    assert np.isnan(quat3._coord2) == True
    # q0p3 + q3p0 + q1p2 − q2p1
    assert np.isnan(quat3._coord3) == True


    q0, q1, q2, q3 = (1, 2, np.nan, 4)
    r0, r1, r2, r3 = (1, np.nan, 3, 4)
    quat1 = Quaternion(q0, q1, q2, q3)
    quat2 = Quaternion(r0, r1, r2, r3)
    quat3 = quat1.rotate(quat2)
    # q0p0 − q1p1 − q2p2 − q3p3
    assert np.isnan(quat3._coord0) == True
    # q0p1 + q1p0 + q2p3 − q3p2
    assert np.isnan(quat3._coord1) == True
    # q0p2 + q2p0 − q1p3 + q3p1
    assert np.isnan(quat3._coord2) == True
    # q0p3 + q3p0 + q1p2 − q2p1
    assert np.isnan(quat3._coord3) == True


    q0, q1, q2, q3 = (np.nan, np.nan, np.nan, np.nan)
    r0, r1, r2, r3 = (1, 2, 3, 4)
    quat1 = Quaternion(q0, q1, q2, q3)
    quat2 = Quaternion(r0, r1, r2, r3)
    quat3 = quat1.rotate(quat2)
    # q0p0 − q1p1 − q2p2 − q3p3
    assert np.isnan(quat3._coord0) == True
    # q0p1 + q1p0 + q2p3 − q3p2
    assert np.isnan(quat3._coord1) == True
    # q0p2 + q2p0 − q1p3 + q3p1
    assert np.isnan(quat3._coord2) == True
    # q0p3 + q3p0 + q1p2 − q2p1
    assert np.isnan(quat3._coord3) == True

    # Check type error
    q = Quaternion(1, 2, 3, 4)
    with pytest.raises(TypeError):
        q.rotate([1, 2, 3, 4])


def xrotationmatrix(theta):
    rmatrix = np.array([[1, 0, 0],
                        [0, np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                        [0, np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
    return rmatrix


def test_xrotate_vector():
    '''Test xrotate_vector method from Quaternion class.
    '''

    # vector on the xz plane
    coords = [0, 1, 3, 10, 100]
    for x, z in itertools.product(coords, coords):
        # rotate 90
        rotq = Quaternion.xrotate_vector(x, 0, z, 90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == -z
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate -90
        rotq = Quaternion.xrotate_vector(x, 0, z, -90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == z
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate 45
        rotq = Quaternion.xrotate_vector(x, 0, z, 45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == -z / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == z / np.sqrt(2)
        # rotate -45
        rotq = Quaternion.xrotate_vector(x, 0, z, -45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == z / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == z / np.sqrt(2)
        # rotate 180
        rotq = Quaternion.xrotate_vector(x, 0, z, 180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == -z
        # rotate -180
        rotq = Quaternion.xrotate_vector(x, 0, z, -180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == -z
        # rotate 135
        rotq = Quaternion.xrotate_vector(x, 0, z, 135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == -z / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == -z / np.sqrt(2)
        # rotate -135
        rotq = Quaternion.xrotate_vector(x, 0, z, -135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == z / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == -z / np.sqrt(2)

    # vector on the yz plane
    for y, z in itertools.product(coords, coords):
        # rotate 90
        rotq = Quaternion.xrotate_vector(0, y, z, 90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == -z
        assert pytest.approx(rotq._coord3, 0.001) == y
        # rotate -90
        rotq = Quaternion.xrotate_vector(0, y, z, -90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == z
        assert pytest.approx(rotq._coord3, 0.001) == -y

    # vector on the xy plane
    for x, y in itertools.product(coords, coords):
        # rotate 90
        rotq = Quaternion.xrotate_vector(x, y, 0, 90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == y
        # rotate -90
        rotq = Quaternion.xrotate_vector(x, y, 0, -90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == -y
        # rotate 180
        rotq = Quaternion.xrotate_vector(x, y, 0, 180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == -y
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate -180
        rotq = Quaternion.xrotate_vector(x, y, 0, -180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == -y
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate 45
        rotq = Quaternion.xrotate_vector(x, y, 0, 45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == y / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == y / np.sqrt(2)
        # rotate -45
        rotq = Quaternion.xrotate_vector(x, y, 0, -45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == y / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == -y / np.sqrt(2)
        # rotate 135
        rotq = Quaternion.xrotate_vector(x, y, 0, 135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == -y / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == y / np.sqrt(2)
        # rotate -135
        rotq = Quaternion.xrotate_vector(x, y, 0, -135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == -y / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == -y / np.sqrt(2)

    # geneal vector
    coords = [-45, -2, 0, 1, 3, 10, 100]
    for x, y, z in itertools.product(coords, coords, coords):
        for theta in np.linspace(-180, 180, 18*4+1):
            rotq = Quaternion.xrotate_vector(x, y, z, theta)
            expected = xrotationmatrix(theta).dot(np.array([x, y, z]))
            assert pytest.approx(rotq._coord0, 0.001) == 0
            assert pytest.approx(rotq._coord1, 0.001) == expected[0]
            assert pytest.approx(rotq._coord2, 0.001) == expected[1]
            assert pytest.approx(rotq._coord3, 0.001) == expected[2]

    rotq = Quaternion.xrotate_vector(1, 1, np.nan, 90)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True

    rotq = Quaternion.xrotate_vector(1, 1, 3, np.nan)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True

    rotq = Quaternion.xrotate_vector(1, np.nan, 3, np.nan)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True


def yrotationmatrix(theta):
    rmatrix = np.array([[np.cos(np.radians(theta)), 0, np.sin(np.radians(theta))],
                        [0,  1, 0],
                        [-np.sin(np.radians(theta)), 0, np.cos(np.radians(theta))]])
    return rmatrix


def test_yrotate_vector():
    '''Test yrotate_vector method from Quaternion class.
    '''

    # vector on the xz plane
    coords = [0, 1, 3, 10, 100]
    for x, z in itertools.product(coords, coords):
        # rotate 90
        rotq = Quaternion.yrotate_vector(x, 0, z, 90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == z
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == -x
        # rotate -90
        rotq = Quaternion.yrotate_vector(x, 0, z, -90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -z
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == x
        # rotate 180
        rotq = Quaternion.yrotate_vector(x, 0, z, 180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == -z
        # rotate -180
        rotq = Quaternion.yrotate_vector(x, 0, z, -180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == -z

    # vector on the yz plane
    coords = [0, 1, 3, 10, 100]
    for y, z in itertools.product(coords, coords):
        # rotate 90
        rotq = Quaternion.yrotate_vector(0, y, z, 90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == z
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate -90
        rotq = Quaternion.yrotate_vector(0, y, z, -90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -z
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate 180
        rotq = Quaternion.yrotate_vector(0, y, z, 180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == -z
        # rotate -180
        rotq = Quaternion.yrotate_vector(0, y, z, -180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == -z
        # rotate 45
        rotq = Quaternion.yrotate_vector(0, y, z, 45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == z / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == z / np.sqrt(2)
        # rotate -45
        rotq = Quaternion.yrotate_vector(0, y, z, -45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -z / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == z / np.sqrt(2)
        # rotate 135
        rotq = Quaternion.yrotate_vector(0, y, z, 135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == z / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == -z / np.sqrt(2)
        # rotate -135
        rotq = Quaternion.yrotate_vector(0, y, z, -135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -z / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == -z / np.sqrt(2)

    # vector on the xy plane
    coords = [0, 1, 3, 10, 100]
    for x, y in itertools.product(coords, coords):
        # rotate 90
        rotq = Quaternion.yrotate_vector(x, y, 0, 90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == -x
        # rotate -90
        rotq = Quaternion.yrotate_vector(x, y, 0, -90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == x
        # rotate 45
        rotq = Quaternion.yrotate_vector(x, y, 0, 45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == -x / np.sqrt(2)
        # rotate -45
        rotq = Quaternion.yrotate_vector(x, y, 0, -45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == x / np.sqrt(2)
        # rotate 180
        rotq = Quaternion.yrotate_vector(x, y, 0, 180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate -180
        rotq = Quaternion.yrotate_vector(x, y, 0, -180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate 135
        rotq = Quaternion.yrotate_vector(x, y, 0, 135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == -x / np.sqrt(2)
        # rotate -135
        rotq = Quaternion.yrotate_vector(x, y, 0, -135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == x / np.sqrt(2)

    # geneal vector
    coords = [-45, -2, 0, 1, 3, 10, 100]
    for x, y, z in itertools.product(coords, coords, coords):
        for theta in np.linspace(-180, 180, 18*4+1):
            rotq = Quaternion.yrotate_vector(x, y, z, theta)
            expected = yrotationmatrix(theta).dot(np.array([x, y, z]))
            assert pytest.approx(rotq._coord0, 0.001) == 0
            assert pytest.approx(rotq._coord1, 0.001) == expected[0]
            assert pytest.approx(rotq._coord2, 0.001) == expected[1]
            assert pytest.approx(rotq._coord3, 0.001) == expected[2]

    rotq = Quaternion.yrotate_vector(1, 1, np.nan, 90)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True

    rotq = Quaternion.yrotate_vector(1, 1, 3, np.nan)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True

    rotq = Quaternion.yrotate_vector(1, np.nan, 3, np.nan)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True


def zrotationmatrix(theta):
    rmatrix = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 0],
                        [np.sin(np.radians(theta)), np.cos(np.radians(theta)), 0],
                        [0, 0, 1]])
    return rmatrix


def test_zrotate_vector():
    '''Test yrotate_vector method from Quaternion class.
    '''
    # vector on the xz plane
    coords = [0, 1, 3, 10, 100]
    for x, z in itertools.product(coords, coords):
        # rotate 90
        rotq = Quaternion.zrotate_vector(x, 0, z, 90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == x
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate -90
        rotq = Quaternion.zrotate_vector(x, 0, z, -90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == -x
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate 45
        rotq = Quaternion.zrotate_vector(x, 0, z, 45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == x / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate -45
        rotq = Quaternion.zrotate_vector(x, 0, z, -45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == -x / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate 180
        rotq = Quaternion.zrotate_vector(x, 0, z, 180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate -180
        rotq = Quaternion.zrotate_vector(x, 0, z, -180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate 135
        rotq = Quaternion.zrotate_vector(x, 0, z, 135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == x / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate -135
        rotq = Quaternion.zrotate_vector(x, 0, z, -135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == -x / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == z

    # vector on the xy plane
    coords = [0, 1, 3, 10, 100]
    for x, y in itertools.product(coords, coords):
        # rotate 90
        rotq = Quaternion.zrotate_vector(x, y, 0, 90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -y
        assert pytest.approx(rotq._coord2, 0.001) == x
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate -90
        rotq = Quaternion.zrotate_vector(x, y, 0, -90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == y
        assert pytest.approx(rotq._coord2, 0.001) == -x
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate 180
        rotq = Quaternion.zrotate_vector(x, y, 0, 180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x
        assert pytest.approx(rotq._coord2, 0.001) == -y
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate -180
        rotq = Quaternion.zrotate_vector(x, y, 0, -180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x
        assert pytest.approx(rotq._coord2, 0.001) == -y
        assert pytest.approx(rotq._coord3, 0.001) == 0

    # vector on the yz plane
    coords = [0, 1, 3, 10, 100]
    for y, z in itertools.product(coords, coords):
        # rotate 90
        rotq = Quaternion.zrotate_vector(0, y, z, 90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -y
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate -90
        rotq = Quaternion.zrotate_vector(0, y, z, -90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == y
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate 45
        rotq = Quaternion.zrotate_vector(0, y, z, 45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -y / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == y / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate -45
        rotq = Quaternion.zrotate_vector(0, y, z, -45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == y / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == y / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate 180
        rotq = Quaternion.zrotate_vector(0, y, z, 180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == -y
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate -180
        rotq = Quaternion.zrotate_vector(0, y, z, -180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == -y
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate 135
        rotq = Quaternion.zrotate_vector(0, y, z, 135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -y / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == -y / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate -135
        rotq = Quaternion.zrotate_vector(0, y, z, -135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == y / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == -y / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == z

    # geneal vector
    coords = [-45, -2, 0, 1, 3, 10, 100]
    for x, y, z in itertools.product(coords, coords, coords):
        for theta in np.linspace(-180, 180, 18*4+1):
            rotq = Quaternion.zrotate_vector(x, y, z, theta)
            expected = zrotationmatrix(theta).dot(np.array([x, y, z]))
            assert pytest.approx(rotq._coord0, 0.001) == 0
            assert pytest.approx(rotq._coord1, 0.001) == expected[0]
            assert pytest.approx(rotq._coord2, 0.001) == expected[1]
            assert pytest.approx(rotq._coord3, 0.001) == expected[2]

    rotq = Quaternion.zrotate_vector(1, 1, np.nan, 90)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True

    rotq = Quaternion.zrotate_vector(1, 1, 3, np.nan)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True

    rotq = Quaternion.yrotate_vector(1, np.nan, 3, np.nan)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True


def test_rotate_vector_along():
    '''Test rotate_vector_along method from Quaternion class.
    '''

    # rotate along x-axis
    coords = [-45, -2, 0, 1, 3, 10, 100]
    for x, y, z in itertools.product(coords, coords, coords):
        for theta in np.linspace(-180, 180, 18*4+1):
            rotq = Quaternion.rotate_vector_along(x, y, z, 1, 0, 0, theta)
            expected = xrotationmatrix(theta).dot(np.array([x, y, z]))
            assert pytest.approx(rotq._coord0, 0.001) == 0
            assert pytest.approx(rotq._coord1, 0.001) == expected[0]
            assert pytest.approx(rotq._coord2, 0.001) == expected[1]
            assert pytest.approx(rotq._coord3, 0.001) == expected[2]
    
    # rotate along y-axis
    coords = [-45, -2, 0, 1, 3, 10, 100]
    for x, y, z in itertools.product(coords, coords, coords):
        for theta in np.linspace(-180, 180, 18*4+1):
            rotq = Quaternion.rotate_vector_along(x, y, z, 0, 1, 0, theta)
            expected = yrotationmatrix(theta).dot(np.array([x, y, z]))
            assert pytest.approx(rotq._coord0, 0.001) == 0
            assert pytest.approx(rotq._coord1, 0.001) == expected[0]
            assert pytest.approx(rotq._coord2, 0.001) == expected[1]
            assert pytest.approx(rotq._coord3, 0.001) == expected[2]
    
    # rotate along z-axis
    coords = [-45, -2, 0, 1, 3, 10, 100]
    for x, y, z in itertools.product(coords, coords, coords):
        for theta in np.linspace(-180, 180, 18*4+1):
            rotq = Quaternion.rotate_vector_along(x, y, z, 0, 0, 1, theta)
            expected = zrotationmatrix(theta).dot(np.array([x, y, z]))
            assert pytest.approx(rotq._coord0, 0.001) == 0
            assert pytest.approx(rotq._coord1, 0.001) == expected[0]
            assert pytest.approx(rotq._coord2, 0.001) == expected[1]
            assert pytest.approx(rotq._coord3, 0.001) == expected[2]

    
    # test normalization of direction vector
    coords = [-45, 0, 1, 10]
    for x, y, z in itertools.product(coords, coords, coords):
        for theta in np.linspace(-180, 180, 18*4+1):
            for direction in [1, 2, 4.4123, 123.4]:
                rotq = Quaternion.rotate_vector_along(x, y, z, direction, 0, 0, theta)
                expected = xrotationmatrix(theta).dot(np.array([x, y, z]))
                assert pytest.approx(rotq._coord0, 0.001) == 0
                assert pytest.approx(rotq._coord1, 0.001) == expected[0]
                assert pytest.approx(rotq._coord2, 0.001) == expected[1]
                assert pytest.approx(rotq._coord3, 0.001) == expected[2]
    

    rotq = Quaternion.rotate_vector_along(1, np.nan, 3, 1, 0, 0, 15)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True

    rotq = Quaternion.rotate_vector_along(1, 0, 3, 1, np.nan, 0, 15)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True

    rotq = Quaternion.rotate_vector_along(1, 2, 3, 1, 0, 0, np.nan)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True

    rotq = Quaternion.rotate_vector_along(1, np.nan, 3, 1, 0, 0, np.nan)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True

    rotq = Quaternion.rotate_vector_along(1, np.nan, 3, 1, np.nan, 0, 15)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True

    rotq = Quaternion.rotate_vector_along(1, np.nan, np.nan, 1, np.nan, np.nan, np.nan)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True


def test_xrotate_reference():
    '''Test xrotate_reference method from Quaternion class.
    '''

    # vector on the xz plane
    coords = [0, 1, 3, 10, 100]
    for x, z in itertools.product(coords, coords):
        # rotate 90
        rotq = Quaternion.xrotate_reference(x, 0, z, 90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == z
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate -90
        rotq = Quaternion.xrotate_reference(x, 0, z, -90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == -z
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate 45
        rotq = Quaternion.xrotate_reference(x, 0, z, 45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == z / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == z / np.sqrt(2)
        # rotate -45
        rotq = Quaternion.xrotate_reference(x, 0, z, -45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == -z / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == z / np.sqrt(2)
        # rotate 180
        rotq = Quaternion.xrotate_reference(x, 0, z, 180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == -z
        # rotate -180
        rotq = Quaternion.xrotate_reference(x, 0, z, -180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == -z
        # rotate 135
        rotq = Quaternion.xrotate_reference(x, 0, z, 135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == z / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == -z / np.sqrt(2)
        # rotate -135
        rotq = Quaternion.xrotate_reference(x, 0, z, -135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == -z / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == -z / np.sqrt(2)
    
    # vector on the xy plane
    coords = [0, 1, 3, 10, 100]
    for x, y in itertools.product(coords, coords):
        # rotate 90
        rotq = Quaternion.xrotate_reference(x, y, 0, 90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == -y
        # rotate -90
        rotq = Quaternion.xrotate_reference(x, y, 0, -90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == y
        # rotate 45
        rotq = Quaternion.xrotate_reference(x, y, 0, 45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == y / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == -y / np.sqrt(2)
        # rotate -45
        rotq = Quaternion.xrotate_reference(x, y, 0, -45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == y / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == y / np.sqrt(2)
        # rotate 180
        rotq = Quaternion.xrotate_reference(x, y, 0, 180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == -y
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate -180
        rotq = Quaternion.xrotate_reference(x, y, 0, -180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == -y
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate 135
        rotq = Quaternion.xrotate_reference(x, y, 0, 135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == -y / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == -y / np.sqrt(2)
        # rotate -135
        rotq = Quaternion.xrotate_reference(x, y, 0, -135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x
        assert pytest.approx(rotq._coord2, 0.001) == -y / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == y / np.sqrt(2)
    
    # vector on the yz plane
    coords = [0, 1, 3, 10, 100]
    for y, z in itertools.product(coords, coords):
        # rotate 90
        rotq = Quaternion.xrotate_reference(0, y, z, 90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == z
        assert pytest.approx(rotq._coord3, 0.001) == -y
        # rotate -90
        rotq = Quaternion.xrotate_reference(0, y, z, -90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == -z
        assert pytest.approx(rotq._coord3, 0.001) == y
        # rotate 180
        rotq = Quaternion.xrotate_reference(0, y, z, 180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == -y
        assert pytest.approx(rotq._coord3, 0.001) == -z
        # rotate -180
        rotq = Quaternion.xrotate_reference(0, y, z, -180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == -y
        assert pytest.approx(rotq._coord3, 0.001) == -z
    
    # geneal vector
    coords = [-45, -2, 0, 1, 3, 10, 100]
    for x, y, z in itertools.product(coords, coords, coords):
        for theta in np.linspace(-180, 180, 18*4+1):
            rotq = Quaternion.xrotate_reference(x, y, z, theta)
            expected = xrotationmatrix(-theta).dot(np.array([x, y, z]))
            assert pytest.approx(rotq._coord0, 0.001) == 0
            assert pytest.approx(rotq._coord1, 0.001) == expected[0]
            assert pytest.approx(rotq._coord2, 0.001) == expected[1]
            assert pytest.approx(rotq._coord3, 0.001) == expected[2]
    
    rotq = Quaternion.xrotate_reference(1, 1, np.nan, 90)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True

    rotq = Quaternion.xrotate_reference(1, 1, 3, np.nan)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True

    rotq = Quaternion.xrotate_reference(1, np.nan, 3, np.nan)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True


def test_yrotate_reference():
    '''Test yrotate_reference method from Quaternion class.
    '''

    # vector on the xz plane
    coords = [0, 1, 3, 10, 100]
    for x, z in itertools.product(coords, coords):
        # rotate 90
        rotq = Quaternion.yrotate_reference(x, 0, z, 90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -z
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == x
        # rotate -90
        rotq = Quaternion.yrotate_reference(x, 0, z, -90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == z
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == -x
        # rotate 180
        rotq = Quaternion.yrotate_reference(x, 0, z, 180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == -z
        # rotate -180
        rotq = Quaternion.yrotate_reference(x, 0, z, -180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == -z

    # vector on the xy plane
    coords = [0, 1, 3, 10, 100]
    for x, y in itertools.product(coords, coords):
        # rotate 90
        rotq = Quaternion.yrotate_reference(x, y, 0, 90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == x
        # rotate -90
        rotq = Quaternion.yrotate_reference(x, y, 0, -90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == -x
        # rotate 45
        rotq = Quaternion.yrotate_reference(x, y, 0, 45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == x / np.sqrt(2)
        # rotate -45
        rotq = Quaternion.yrotate_reference(x, y, 0, -45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == -x / np.sqrt(2)
        # rotate 180
        rotq = Quaternion.yrotate_reference(x, y, 0, 180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate -180
        rotq = Quaternion.yrotate_reference(x, y, 0, -180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate 135
        rotq = Quaternion.yrotate_reference(x, y, 0, 45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == x / np.sqrt(2)
        # rotate -135
        rotq = Quaternion.yrotate_reference(x, y, 0, -45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == -x / np.sqrt(2)
    
    # vector on the yz plane
    coords = [0, 1, 3, 10, 100]
    for y, z in itertools.product(coords, coords):
        # rotate 90
        rotq = Quaternion.yrotate_reference(0, y, z, 90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -z
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate -90
        rotq = Quaternion.yrotate_reference(0, y, z, -90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == z
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate 45
        rotq = Quaternion.yrotate_reference(0, y, z, 45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -z / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == z / np.sqrt(2)
        # rotate -45
        rotq = Quaternion.yrotate_reference(0, y, z, -45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == z / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == z / np.sqrt(2)
        # rotate 180
        rotq = Quaternion.yrotate_reference(0, y, z, 180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == -z
        # rotate -180
        rotq = Quaternion.yrotate_reference(0, y, z, -180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == -z
        # rotate 135
        rotq = Quaternion.yrotate_reference(0, y, z, 135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -z / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == -z / np.sqrt(2)
        # rotate -135
        rotq = Quaternion.yrotate_reference(0, y, z, -135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == z / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == y
        assert pytest.approx(rotq._coord3, 0.001) == -z / np.sqrt(2)
    
    # geneal vector
    coords = [-45, -2, 0, 1, 3, 10, 100]
    for x, y, z in itertools.product(coords, coords, coords):
        for theta in np.linspace(-180, 180, 18*4+1):
            rotq = Quaternion.yrotate_reference(x, y, z, theta)
            expected = yrotationmatrix(-theta).dot(np.array([x, y, z]))
            assert pytest.approx(rotq._coord0, 0.001) == 0
            assert pytest.approx(rotq._coord1, 0.001) == expected[0]
            assert pytest.approx(rotq._coord2, 0.001) == expected[1]
            assert pytest.approx(rotq._coord3, 0.001) == expected[2]

    rotq = Quaternion.yrotate_reference(1, 1, np.nan, 90)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True

    rotq = Quaternion.yrotate_reference(1, 1, 3, np.nan)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True

    rotq = Quaternion.yrotate_reference(1, np.nan, 3, np.nan)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True


def test_zrotate_reference():
    '''Test zrotate_reference method from Quaternion class.
    '''

    # vector on the xz plane
    coords = [0, 1, 3, 10, 100]
    for x, z in itertools.product(coords, coords):
        # rotate 90
        rotq = Quaternion.zrotate_reference(x, 0, z, 90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == -x
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate -90
        rotq = Quaternion.zrotate_reference(x, 0, z, -90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == x
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate 45
        rotq = Quaternion.zrotate_reference(x, 0, z, 45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == -x / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate -45
        rotq = Quaternion.zrotate_reference(x, 0, z, -45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == x / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == x / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate 180
        rotq = Quaternion.zrotate_reference(x, 0, z, 180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate -180
        rotq = Quaternion.zrotate_reference(x, 0, z, -180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate 135
        rotq = Quaternion.zrotate_reference(x, 0, z, 135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == -x / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate -135
        rotq = Quaternion.zrotate_reference(x, 0, z, -135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == x / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == z
    
    # vector on the xy plane
    coords = [0, 1, 3, 10, 100]
    for x, y in itertools.product(coords, coords):
        # rotate 90
        rotq = Quaternion.zrotate_reference(x, y, 0, 90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == y
        assert pytest.approx(rotq._coord2, 0.001) == -x
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate -90
        rotq = Quaternion.zrotate_reference(x, y, 0, -90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -y
        assert pytest.approx(rotq._coord2, 0.001) == x
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate 180
        rotq = Quaternion.zrotate_reference(x, y, 0, 180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x
        assert pytest.approx(rotq._coord2, 0.001) == -y
        assert pytest.approx(rotq._coord3, 0.001) == 0
        # rotate -180
        rotq = Quaternion.zrotate_reference(x, y, 0, -180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -x
        assert pytest.approx(rotq._coord2, 0.001) == -y
        assert pytest.approx(rotq._coord3, 0.001) == 0
    
    # vector on the yz plane
    coords = [0, 1, 3, 10, 100]
    for y, z in itertools.product(coords, coords):
        # rotate 90
        rotq = Quaternion.zrotate_reference(0, y, z, 90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == y
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate -90
        rotq = Quaternion.zrotate_reference(0, y, z, -90)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -y
        assert pytest.approx(rotq._coord2, 0.001) == 0
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate 45
        rotq = Quaternion.zrotate_reference(0, y, z, 45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == y / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == y / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate -45
        rotq = Quaternion.zrotate_reference(0, y, z, -45)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -y / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == y / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate 180
        rotq = Quaternion.zrotate_reference(0, y, z, 180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == -y
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate -180
        rotq = Quaternion.zrotate_reference(0, y, z, -180)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == 0
        assert pytest.approx(rotq._coord2, 0.001) == -y
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate 135
        rotq = Quaternion.zrotate_reference(0, y, z, 135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == y / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == -y / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == z
        # rotate -135
        rotq = Quaternion.zrotate_reference(0, y, z, -135)
        assert pytest.approx(rotq._coord0, 0.001) == 0
        assert pytest.approx(rotq._coord1, 0.001) == -y / np.sqrt(2)
        assert pytest.approx(rotq._coord2, 0.001) == -y / np.sqrt(2)
        assert pytest.approx(rotq._coord3, 0.001) == z
    
    # geneal vector
    coords = [-45, -2, 0, 1, 3, 10, 100]
    for x, y, z in itertools.product(coords, coords, coords):
        for theta in np.linspace(-180, 180, 18*4+1):
            rotq = Quaternion.zrotate_reference(x, y, z, theta)
            expected = zrotationmatrix(-theta).dot(np.array([x, y, z]))
            assert pytest.approx(rotq._coord0, 0.001) == 0
            assert pytest.approx(rotq._coord1, 0.001) == expected[0]
            assert pytest.approx(rotq._coord2, 0.001) == expected[1]
            assert pytest.approx(rotq._coord3, 0.001) == expected[2]

    rotq = Quaternion.zrotate_reference(1, 1, np.nan, 90)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True

    rotq = Quaternion.zrotate_reference(1, 1, 3, np.nan)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True

    rotq = Quaternion.zrotate_reference(1, np.nan, 3, np.nan)
    assert np.isnan(rotq._coord0) == True
    assert np.isnan(rotq._coord1) == True
    assert np.isnan(rotq._coord2) == True
    assert np.isnan(rotq._coord3) == True