package Persons.impl;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;
import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;
import org.eclipse.emf.ecore.impl.PackageImpl;

import Persons.PersonsPackage;
import Persons.PersonsFactory;
import Persons.Person;
import Persons.Male;
import Persons.Female;
import Persons.PersonRegister;

/**
 * <!-- begin-user-doc -->
 * An implementation of the package '<em><b>Persons Package</b></em>'.
 * <!-- end-user-doc -->
 *
 * @generated
 */
public class PersonsPackageImpl extends PackageImpl implements PersonsPackage {
	/**
	 * @generated
	 */
	public static final int PERSON_REGISTER__PERSONS = 0;

	/**
	 * @generated
	 */
	public static final int PERSON_REGISTER_FEATURE_COUNT = 1;

	/**
	 * @generated
	 */
	public static final int PERSON__NAME = 0;

	/**
	 * @generated
	 */
	public static final int PERSON__BIRTHDAY = 1;

	/**
	 * @generated
	 */
	public static final int PERSON__PERSONS_INVERSE = 2;

	/**
	 * @generated
	 */
	public static final int PERSON_FEATURE_COUNT = 3;

	/**
	 * @generated
	 */
	public static final int MALE_FEATURE_COUNT = PERSON_FEATURE_COUNT;

	/**
	 * @generated
	 */
	public static final int FEMALE_FEATURE_COUNT = PERSON_FEATURE_COUNT;

	/**
	 * @generated
	 */
	public static PersonsPackageImpl init() {
		return thePersonsPackage;
	}

	/**
	 * @generated
	 */
	private static PersonsPackageImpl thePersonsPackage;

	/**
	 * @generated
	 */
	protected PersonsPackageImpl() {
		super();
	}

	/**
	 * @generated
	 */
	@Override
	public EClass getPersonRegister() {
		return Literals.PERSON_REGISTER;
	}

	/**
	 * @generated
	 */
	@Override
	public EReference getPersonRegister_Persons() {
		return Literals.PERSON_REGISTER__PERSONS;
	}

	/**
	 * @generated
	 */
	@Override
	public EClass getPerson() {
		return Literals.PERSON;
	}

	/**
	 * @generated
	 */
	@Override
	public EAttribute getPerson_Name() {
		return Literals.PERSON__NAME;
	}

	/**
	 * @generated
	 */
	@Override
	public EAttribute getPerson_Birthday() {
		return Literals.PERSON__BIRTHDAY;
	}

	/**
	 * @generated
	 */
	@Override
	public EReference getPerson_PersonsInverse() {
		return Literals.PERSON__PERSONS_INVERSE;
	}

	/**
	 * @generated
	 */
	@Override
	public EClass getMale() {
		return Literals.MALE;
	}

	/**
	 * @generated
	 */
	@Override
	public EClass getFemale() {
		return Literals.FEMALE;
	}

	/**
	 * @generated
	 */
	@Override
	public EFactory getEFactoryInstance() {
		return PersonsFactoryImpl.init();
	}

	/**
	 * @generated
	 */
	protected EClass personRegisterEClass = null;

	/**
	 * @generated
	 */
	protected EReference personRegister__PersonsEClass = null;

	/**
	 * @generated
	 */
	protected EClass personEClass = null;

	/**
	 * @generated
	 */
	protected EAttribute person__NameEAttribute = null;

	/**
	 * @generated
	 */
	protected EAttribute person__BirthdayEAttribute = null;

	/**
	 * @generated
	 */
	protected EReference person__PersonsInverseEReference = null;

	/**
	 * @generated
	 */
	protected EClass maleEClass = null;

	/**
	 * @generated
	 */
	protected EClass femaleEClass = null;

	/**
	 * @generated
	 */
	@Override
	public void createPackageContents() {
		// Initialize package
		setnsURI("http://persons");
		setnsPrefix("Persons");
		
		// Create PersonRegister class
		personRegisterEClass = createEClass("PersonRegister", false);
		personRegister__PersonsEClass = createEReference("PersonRegister", "Persons", personRegisterEClass, PERSON_REGISTER__PERSONS, 0, -1, Person.class, true);
		
		// Create Person class (abstract)
		personEClass = createEClass("Person", true);
		person__NameEAttribute = createEAttribute("Person", "Name", personEClass, PERSON__NAME, 0, 1, String.class, false);
		person__BirthdayEAttribute = createEAttribute("Person", "Birthday", personEClass, PERSON__BIRTHDAY, 0, 1, Date.class, false);
		person__PersonsInverseEReference = createEReference("Person", "PersonsInverse", personEClass, PERSON__PERSONS_INVERSE, 0, 1, PersonRegister.class, false);
		
		// Create Male class
		maleEClass = createEClass("Male", false);
		
		// Create Female class
		femaleEClass = createEClass("Female", false);
	}

	/**
	 * @generated
	 */
	public EClass createEClass(String name, boolean abstract_) {
		EClass eClass = new MinimalEObjectImpl.Container() {};
		eClass.setClassifierID(0);
		eClass.setName(name);
		eClass.setAbstract(abstract_);
		eClass.setInstanceClassName("Persons." + name);
		return eClass;
	}

	/**
	 * @generated
	 */
	public EAttribute createEAttribute(String className, String attrName, EClass eClass, int featureID, int lowerBound, int upperBound, Class<?> instanceClass, boolean unique) {
		EAttribute eAttribute = super.createEAttribute(eClass, featureID);
		eAttribute.setName(attrName);
		eAttribute.setEType(instanceClass);
		eAttribute.setLowerBound(lowerBound);
		eAttribute.setUpperBound(upperBound);
		eAttribute.setUnique(unique);
		return eAttribute;
	}

	/**
	 * @generated
	 */
	public EReference createEReference(String className, String refName, EClass eClass, int featureID, int lowerBound, int upperBound, Class<?> instanceClass, boolean containment) {
		EReference eReference = super.createEReference(eClass, featureID);
		eReference.setName(refName);
		eReference.setEType(instanceClass);
		eReference.setLowerBound(lowerBound);
		eReference.setUpperBound(upperBound);
		eReference.setContainment(containment);
		return eReference;
	}

	/**
	 * @generated
	 */
	public static final class Literals {
		public static final EClass PERSON_REGISTER = thePersonsPackage != null ? thePersonsPackage.personRegisterEClass : (thePersonsPackage.personRegisterEClass = new EClassImpl());
		public static final EReference PERSON_REGISTER__PERSONS = thePersonsPackage != null ? thePersonsPackage.personRegister__PersonsEClass : (thePersonsPackage.personRegister__PersonsEClass = new EReferenceImpl());
		public static final EClass PERSON = thePersonsPackage != null ? thePersonsPackage.personEClass : (thePersonsPackage.personEClass = new EClassImpl());
		public static final EAttribute PERSON__NAME = thePersonsPackage != null ? thePersonsPackage.person__NameEAttribute : (thePersonsPackage.person__NameEAttribute = new EAttributeImpl());
		public static final EAttribute PERSON__BIRTHDAY = thePersonsPackage != null ? thePersonsPackage.person__BirthdayEAttribute : (thePersonsPackage.person__BirthdayEAttribute = new EAttributeImpl());
		public static final EReference PERSON__PERSONS_INVERSE = thePersonsPackage != null ? thePersonsPackage.person__PersonsInverseEReference : (thePersonsPackage.person__PersonsInverseEReference = new EReferenceImpl());
		public static final EClass MALE = thePersonsPackage != null ? thePersonsPackage.maleEClass : (thePersonsPackage.maleEClass = new EClassImpl());
		public static final EClass FEMALE = thePersonsPackage != null ? thePersonsPackage.femaleEClass : (thePersonsPackage.femaleEClass = new EClassImpl());
	}

	/**
	 * @generated
	 */
	private static class EClassImpl extends MinimalEObjectImpl implements EClass {
	}

	/**
	 * @generated
	 */
	private static class EAttributeImpl extends MinimalEObjectImpl implements EAttribute {
	}

	/**
	 * @generated
	 */
	private static class EReferenceImpl extends MinimalEObjectImpl implements EReference {
	}
} // PersonsPackageImpl