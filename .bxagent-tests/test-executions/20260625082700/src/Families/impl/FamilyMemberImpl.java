package Families.impl;

import java.util.Date;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.BasicEObjectImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Family Member</b></em>'.
 * <!-- end-user-doc -->
 *
 * @generated
 */
public class FamilyMemberImpl extends BasicEObjectImpl implements FamilyMember {
	/**
	 * @generated
	 */
	protected FamilyMemberImpl() {
		super();
	}

	/**
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return FamiliesPackageImpl.Literals.FAMILY_MEMBER;
	}

	/**
	 * @generated
	 */
	protected String name;

	/**
	 * @generated
	 */
	@Override
	public String getName() {
		return name;
	}

	/**
	 * @generated
	 */
	@Override
	public void setName(String newName) {
		name = newName;
	}

	/**
	 * @generated
	 */
	protected Family fatherInverse;

	/**
	 * @generated
	 */
	@Override
	public Family getFatherInverse() {
		if (fatherInverse != null && fatherInverse.eIsProxy()) {
			InternalEObject o = (InternalEObject) fatherInverse;
			fatherInverse = (Family) eResolveProxy(o);
		}
		return fatherInverse;
	}

	/**
	 * @generated
	 */
	@Override
	public Family basicSetFatherInverse(Family newFatherInverse, Notifications msgs) {
		Family oldFatherInverse = fatherInverse;
		fatherInverse = newFatherInverse;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notifications.SET, FamiliesPackageImpl.FAMILY_MEMBER__FATHER_INVERSE, oldFatherInverse, fatherInverse);
			if (msgs == null) {
				deliverChangeNotification(notification);
			} else {
				msgs.add(notification);
			}
		}
		return oldFatherInverse;
	}

	/**
	 * @generated
	 */
	@Override
	public void setFatherInverse(Family newFatherInverse) {
		setFatherInverseInternal(newFatherInverse);
	}

	/**
	 * @generated
	 */
	public void setFatherInverseInternal(Family newFatherInverse) {
		Family oldFatherInverse = fatherInverse;
		fatherInverse = newFatherInverse;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notifications.SET, FamiliesPackageImpl.FAMILY_MEMBER__FATHER_INVERSE, oldFatherInverse, fatherInverse);
			if (oldFatherInverse != null) {
				notification.setOldValue(oldFatherInverse);
			}
			if (newFatherInverse != null) {
				notification.setNewValue(newFatherInverse);
			}
			eNotify(notification);
		}
	}

	/**
	 * @generated
	 */
	protected Family motherInverse;

	/**
	 * @generated
	 */
	@Override
	public Family getMotherInverse() {
		if (motherInverse != null && motherInverse.eIsProxy()) {
			InternalEObject o = (InternalEObject) motherInverse;
			motherInverse = (Family) eResolveProxy(o);
		}
		return motherInverse;
	}

	/**
	 * @generated
	 */
	@Override
	public Family basicSetMotherInverse(Family newMotherInverse, Notifications msgs) {
		Family oldMotherInverse = motherInverse;
		motherInverse = newMotherInverse;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notifications.SET, FamiliesPackageImpl.FAMILY_MEMBER__MOTHER_INVERSE, oldMotherInverse, motherInverse);
			if (msgs == null) {
				deliverChangeNotification(notification);
			} else {
				msgs.add(notification);
			}
		}
		return oldMotherInverse;
	}

	/**
	 * @generated
	 */
	@Override
	public void setMotherInverse(Family newMotherInverse) {
		setMotherInverseInternal(newMotherInverse);
	}

	/**
	 * @generated
	 */
	public void setMotherInverseInternal(Family newMotherInverse) {
		Family oldMotherInverse = motherInverse;
		motherInverse = newMotherInverse;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notifications.SET, FamiliesPackageImpl.FAMILY_MEMBER__MOTHER_INVERSE, oldMotherInverse, motherInverse);
			if (oldMotherInverse != null) {
				notification.setOldValue(oldMotherInverse);
			}
			if (newMotherInverse != null) {
				notification.setNewValue(newMotherInverse);
			}
			eNotify(notification);
		}
	}

	/**
	 * @generated
	 */
	protected Family sonsInverse;

	/**
	 * @generated
	 */
	@Override
	public Family getSonsInverse() {
		if (sonsInverse != null && sonsInverse.eIsProxy()) {
			InternalEObject o = (InternalEObject) sonsInverse;
			sonsInverse = (Family) eResolveProxy(o);
		}
		return sonsInverse;
	}

	/**
	 * @generated
	 */
	@Override
	public Family basicSetSonsInverse(Family newSonsInverse, Notifications msgs) {
		Family oldSonsInverse = sonsInverse;
		sonsInverse = newSonsInverse;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notifications.SET, FamiliesPackageImpl.FAMILY_MEMBER__SONS_INVERSE, oldSonsInverse, sonsInverse);
			if (msgs == null) {
				deliverChangeNotification(notification);
			} else {
				msgs.add(notification);
			}
		}
		return oldSonsInverse;
	}

	/**
	 * @generated
	 */
	@Override
	public void setSonsInverse(Family newSonsInverse) {
		setSonsInverseInternal(newSonsInverse);
	}

	/**
	 * @generated
	 */
	public void setSonsInverseInternal(Family newSonsInverse) {
		Family oldSonsInverse = sonsInverse;
		sonsInverse = newSonsInverse;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notifications.SET, FamiliesPackageImpl.FAMILY_MEMBER__SONS_INVERSE, oldSonsInverse, sonsInverse);
			if (oldSonsInverse != null) {
				notification.setOldValue(oldSonsInverse);
			}
			if (newSonsInverse != null) {
				notification.setNewValue(newSonsInverse);
			}
			eNotify(notification);
		}
	}

	/**
	 * @generated
	 */
	protected Family daughtersInverse;

	/**
	 * @generated
	 */
	@Override
	public Family getDaughtersInverse() {
		if (daughtersInverse != null && daughtersInverse.eIsProxy()) {
			InternalEObject o = (InternalEObject) daughtersInverse;
			daughtersInverse = (Family) eResolveProxy(o);
		}
		return daughtersInverse;
	}

	/**
	 * @generated
	 */
	@Override
	public Family basicSetDaughtersInverse(Family newDaughtersInverse, Notifications msgs) {
		Family oldDaughtersInverse = daughtersInverse;
		daughtersInverse = newDaughtersInverse;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notifications.SET, FamiliesPackageImpl.FAMILY_MEMBER__DAUGHTERS_INVERSE, oldDaughtersInverse, daughtersInverse);
			if (msgs == null) {
				deliverChangeNotification(notification);
			} else {
				msgs.add(notification);
			}
		}
		return oldDaughtersInverse;
	}

	/**
	 * @generated
	 */
	@Override
	public void setDaughtersInverse(Family newDaughtersInverse) {
		setDaughtersInverseInternal(newDaughtersInverse);
	}

	/**
	 * @generated
	 */
	public void setDaughtersInverseInternal(Family newDaughtersInverse) {
		Family oldDaughtersInverse = daughtersInverse;
		daughtersInverse = newDaughtersInverse;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notifications.SET, FamiliesPackageImpl.FAMILY_MEMBER__DAUGHTERS_INVERSE, oldDaughtersInverse, daughtersInverse);
			if (oldDaughtersInverse != null) {
				notification.setOldValue(oldDaughtersInverse);
			}
			if (newDaughtersInverse != null) {
				notification.setNewValue(newDaughtersInverse);
			}
			eNotify(notification);
		}
	}

	/**
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case FamiliesPackageImpl.FAMILY_MEMBER__NAME:
				return getName();
			case FamiliesPackageImpl.FAMILY_MEMBER__FATHER_INVERSE:
				return getFatherInverse();
			case FamiliesPackageImpl.FAMILY_MEMBER__MOTHER_INVERSE:
				return getMotherInverse();
			case FamiliesPackageImpl.FAMILY_MEMBER__SONS_INVERSE:
				return getSonsInverse();
			case FamiliesPackageImpl.FAMILY_MEMBER__DAUGHTERS_INVERSE:
				return getDaughtersInverse();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * @generated
	 */
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case FamiliesPackageImpl.FAMILY_MEMBER__NAME:
				setName((String) newValue);
				return;
			case FamiliesPackageImpl.FAMILY_MEMBER__FATHER_INVERSE:
				setFatherInverse((Family) newValue);
				return;
			case FamiliesPackageImpl.FAMILY_MEMBER__MOTHER_INVERSE:
				setMotherInverse((Family) newValue);
				return;
			case FamiliesPackageImpl.FAMILY_MEMBER__SONS_INVERSE:
				setSonsInverse((Family) newValue);
				return;
			case FamiliesPackageImpl.FAMILY_MEMBER__DAUGHTERS_INVERSE:
				setDaughtersInverse((Family) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	/**
	 * @generated
	 */
	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case FamiliesPackageImpl.FAMILY_MEMBER__NAME:
				setName(NOT_SET);
				return;
			case FamiliesPackageImpl.FAMILY_MEMBER__FATHER_INVERSE:
				setFatherInverse((Family) NOT_SET);
				return;
			case FamiliesPackageImpl.FAMILY_MEMBER__MOTHER_INVERSE:
				setMotherInverse((Family) NOT_SET);
				return;
			case FamiliesPackageImpl.FAMILY_MEMBER__SONS_INVERSE:
				setSonsInverse((Family) NOT_SET);
				return;
			case FamiliesPackageImpl.FAMILY_MEMBER__DAUGHTERS_INVERSE:
				setDaughtersInverse((Family) NOT_SET);
				return;
		}
		super.eUnset(featureID);
	}

	/**
	 * @generated
	 */
	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case FamiliesPackageImpl.FAMILY_MEMBER__NAME:
				return name != NOT_SET;
			case FamiliesPackageImpl.FAMILY_MEMBER__FATHER_INVERSE:
				return fatherInverse != null;
			case FamiliesPackageImpl.FAMILY_MEMBER__MOTHER_INVERSE:
				return motherInverse != null;
			case FamiliesPackageImpl.FAMILY_MEMBER__SONS_INVERSE:
				return sonsInverse != null;
			case FamiliesPackageImpl.FAMILY_MEMBER__DAUGHTERS_INVERSE:
				return daughtersInverse != null;
		}
		return super.eIsSet(featureID);
	}
} // FamilyMemberImpl