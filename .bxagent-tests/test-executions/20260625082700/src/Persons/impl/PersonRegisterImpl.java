package Persons.impl;

import java.util.Date;
import java.util.ArrayList;
import java.util.List;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;
import org.eclipse.emf.ecore.util.EObjectContainmentWithInverseEList;
import org.eclipse.emf.ecore.util.EObjectWithInverseResolvingEList;
import org.eclipse.emf.ecore.util.InternalEList;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Person Register</b></em>'.
 * <!-- end-user-doc -->
 *
 * @generated
 */
public class PersonRegisterImpl extends MinimalEObjectImpl.Container implements PersonRegister {
	/**
	 * @generated
	 */
	protected PersonRegisterImpl() {
		super();
	}

	/**
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PersonsPackageImpl.Literals.PERSON_REGISTER;
	}

	/**
	 * @generated
	 */
	protected EList<Person> persons;

	/**
	 * @generated
	 */
	@Override
	public EList<Person> getPersons() {
		if (persons == null) {
			persons = new EObjectContainmentWithInverseEList<Person>(Person.class, this, PersonsPackageImpl.PERSON_REGISTER__PERSONS, PersonsPackageImpl.PERSON__PERSONS_INVERSE);
		}
		return persons;
	}

	/**
	 * @generated
	 */
	@Override
	public NotificationChain eInverseAdd(InternalEObject otherEnd, int featureID, Notifications msgs) {
		switch (featureID) {
			case PersonsPackageImpl.PERSON_REGISTER__PERSONS:
				return ((InternalEObject) otherEnd).eInverseAdd(this, PersonsPackageImpl.PERSON__PERSONS_INVERSE, null, msgs);
		}
		return super.eInverseAdd(otherEnd, featureID, msgs);
	}

	/**
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, Notifications msgs) {
		switch (featureID) {
			case PersonsPackageImpl.PERSON_REGISTER__PERSONS:
				return ((InternalEObject) otherEnd).eInverseRemove(this, PersonsPackageImpl.PERSON__PERSONS_INVERSE, null, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	/**
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case PersonsPackageImpl.PERSON_REGISTER__PERSONS:
				if (resolve) {
					return getPersons();
				}
				return getPersons().list();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * @generated
	 */
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case PersonsPackageImpl.PERSON_REGISTER__PERSONS:
				getPersons().clear();
				getPersons().addAll((EList<Person>) newValue);
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
			case PersonsPackageImpl.PERSON_REGISTER__PERSONS:
				getPersons().clear();
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
			case PersonsPackageImpl.PERSON_REGISTER__PERSONS:
				return persons != null && !persons.isEmpty();
		}
		return super.eIsSet(featureID);
	}
} // PersonRegisterImpl