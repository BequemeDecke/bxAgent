package Persons.impl;

import java.util.Date;

import Persons.Male;
import Persons.PersonsPackage;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.util.EObjectContainmentWithInverseEList;
import org.eclipse.emf.ecore.util.EcoreUtil;
import org.eclipse.emf.ecore.util.InternalEList;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Male</b></em>'.
 * <!-- end-user-doc -->
 *
 * @generated
 */
public class MaleImpl extends PersonImpl implements Male {
	/**
	 * @generated
	 */
	protected MaleImpl() {
		super();
	}

	/**
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PersonsPackageImpl.Literals.MALE;
	}

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
	@Override
	public Date getBirthday() {
		return birthday;
	}

	/**
	 * @generated
	 */
	@Override
	public void setBirthday(Date newBirthday) {
		birthday = newBirthday;
	}

	/**
	 * @generated
	 */
	@Override
	public PersonRegister getPersonsInverse() {
		if (personsInverse != null && personsInverse.eIsProxy()) {
			InternalEObject o = (InternalEObject) personsInverse;
			personsInverse = (PersonRegister) eResolveProxy(o);
		}
		return personsInverse;
	}

	/**
	 * @generated
	 */
	@Override
	public PersonRegister basicGetPersonsInverse() {
		return personsInverse;
	}

	/**
	 * @generated
	 */
	@Override
	public void setPersonsInverse(PersonRegister newPersonsInverse) {
		if (newPersonsInverse != null) {
			if (((InternalEObject) newPersonsInverse).eIsProxy()) {
				setPersonsInverse((PersonRegister) eResolveProxy((InternalEObject) newPersonsInverse));
				return;
			}
		}
		if (personsInverse != newPersonsInverse) {
			Notifications msgs = null;
			if (personsInverse != null) {
				msgs = ((InternalEObject) personsInverse).eInverseRemove(this, PersonsPackageImpl.PERSON_REGISTER__PERSONS, null, msgs);
			}
			if (newPersonsInverse != null) {
				msgs = ((InternalEObject) newPersonsInverse).eInverseAdd(this, PersonsPackageImpl.PERSON_REGISTER__PERSONS, null, msgs);
			}
			msgs = basicSetPersonsInverse(newPersonsInverse, msgs);
		} else if (eNotificationRequired()) {
			eNotify(new ENotificationImpl(this, Notifications.SET, PersonsPackageImpl.PERSON__PERSONS_INVERSE, newPersonsInverse, newPersonsInverse));
		}
	}

	/**
	 * @generated
	 */
	@Override
	public NotificationChain eInverseAdd(InternalEObject otherEnd, int featureID, Notifications msgs) {
		switch (featureID) {
			case PersonsPackageImpl.PERSON_REGISTER__PERSONS:
				return basicSetPersonsInverse((PersonRegister) otherEnd, msgs);
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
				return basicSetPersonsInverse(null, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	/**
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case PersonsPackageImpl.PERSON__NAME:
				return getName();
			case PersonsPackageImpl.PERSON__BIRTHDAY:
				return getBirthday();
			case PersonsPackageImpl.PERSON__PERSONS_INVERSE:
				if (resolve) {
					return getPersonsInverse();
				}
				return basicGetPersonsInverse();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * @generated
	 */
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case PersonsPackageImpl.PERSON__NAME:
				setName((String) newValue);
				return;
			case PersonsPackageImpl.PERSON__BIRTHDAY:
				setBirthday((Date) newValue);
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
			case PersonsPackageImpl.PERSON__NAME:
				setName(NOT_SET);
				return;
			case PersonsPackageImpl.PERSON__BIRTHDAY:
				setBirthday(NOT_SET);
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
			case PersonsPackageImpl.PERSON__NAME:
				return name != NOT_SET;
			case PersonsPackageImpl.PERSON__BIRTHDAY:
				return birthday != NOT_SET;
			case PersonsPackageImpl.PERSON__PERSONS_INVERSE:
				return personsInverse != null;
		}
		return super.eIsSet(featureID);
	}
} // MaleImpl