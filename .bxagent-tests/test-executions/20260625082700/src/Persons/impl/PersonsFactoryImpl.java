package Persons.impl;

import Persons.PersonRegister;
import Persons.Person;
import Persons.Male;
import Persons.Female;
import Persons.PersonsFactory;
import Persons.PersonsPackage;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the factory '<em><b>Persons Factory</b></em>'.
 * <!-- end-user-doc -->
 *
 * @generated
 */
public class PersonsFactoryImpl extends MinimalEObjectImpl implements PersonsFactory {
	/**
	 * @generated
	 */
	public static PersonsFactory init() {
		return thePersonsFactory;
	}

	/**
	 * @generated
	 */
	private static PersonsFactory thePersonsFactory;

	/**
	 * @generated
	 */
	protected PersonsFactoryImpl() {
		super();
	}

	/**
	 * @generated
	 */
	@Override
	public EObject create(EClass eClass) {
		switch (eClass.getClassifierID()) {
			case PersonsPackageImpl.Literals.PERSON_REGISTER:
				return createPersonRegister();
			case PersonsPackageImpl.Literals.MALE:
				return createMale();
			case PersonsPackageImpl.Literals.FEMALE:
				return createFemale();
			default:
				throw new IllegalArgumentException("The class '" + eClass.getName() + "' is not a valid classifier");
		}
	}

	/**
	 * @generated
	 */
	@Override
	public PersonRegister createPersonRegister() {
		PersonRegisterImpl personRegister = new PersonRegisterImpl();
		return personRegister;
	}

	/**
	 * @generated
	 */
	@Override
	public Male createMale() {
		MaleImpl male = new MaleImpl();
		return male;
	}

	/**
	 * @generated
	 */
	@Override
	public Female createFemale() {
		FemaleImpl female = new FemaleImpl();
		return female;
	}

	/**
	 * @generated
	 */
	@Override
	public PersonsPackage getPersonsPackage() {
		return PersonsPackageImpl.init();
	}
} // PersonsFactoryImpl