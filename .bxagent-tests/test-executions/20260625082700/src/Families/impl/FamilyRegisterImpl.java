package Families.impl;

import java.util.Date;
import java.util.ArrayList;
import java.util.List;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;
import org.eclipse.emf.ecore.util.EObjectWithInverseResolvingEList;
import org.eclipse.emf.ecore.util.InternalEList;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Family Register</b></em>'.
 * <!-- end-user-doc -->
 *
 * @generated
 */
public class FamilyRegisterImpl extends MinimalEObjectImpl.Container implements FamilyRegister {
	/**
	 * @generated
	 */
	protected FamilyRegisterImpl() {
		super();
	}

	/**
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return FamiliesPackageImpl.Literals.FAMILY_REGISTER;
	}

	/**
	 * @generated
	 */
	protected EList<Family> families;

	/**
	 * @generated
	 */
	@Override
	public EList<Family> getFamilies() {
		if (families == null) {
			families = new EObjectWithInverseResolvingEList<Family>(Family.class, this, FamiliesPackageImpl.FAMILY_REGISTER__FAMILIES, FamiliesPackageImpl.FAMILY__FAMILIES_INVERSE);
		}
		return families;
	}

	/**
	 * @generated
	 */
	@Override
	public NotificationChain eInverseAdd(InternalEObject otherEnd, int featureID, Notifications msgs) {
		switch (featureID) {
			case FamiliesPackageImpl.FAMILY_REGISTER__FAMILIES:
				return ((InternalEObject) otherEnd).eInverseAdd(this, FamiliesPackageImpl.FAMILY__FAMILIES_INVERSE, null, msgs);
		}
		return super.eInverseAdd(otherEnd, featureID, msgs);
	}

	/**
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, Notifications msgs) {
		switch (featureID) {
			case FamiliesPackageImpl.FAMILY_REGISTER__FAMILIES:
				return ((InternalEObject) otherEnd).eInverseRemove(this, FamiliesPackageImpl.FAMILY__FAMILIES_INVERSE, null, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	/**
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case FamiliesPackageImpl.FAMILY_REGISTER__FAMILIES:
				if (resolve) {
					return getFamilies();
				}
				return getFamilies().list();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * @generated
	 */
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case FamiliesPackageImpl.FAMILY_REGISTER__FAMILIES:
				getFamilies().clear();
				getFamilies().addAll((EList<Family>) newValue);
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
			case FamiliesPackageImpl.FAMILY_REGISTER__FAMILIES:
				getFamilies().clear();
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
			case FamiliesPackageImpl.FAMILY_REGISTER__FAMILIES:
				return families != null && !families.isEmpty();
		}
		return super.eIsSet(featureID);
	}
} // FamilyRegisterImpl