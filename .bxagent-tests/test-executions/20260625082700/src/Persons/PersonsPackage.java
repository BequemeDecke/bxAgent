/**
 */
package Persons;

import org.eclipse.emf.ecore.EPackage;

/**
 * <!-- begin-user-doc -->
 * The <b>Package</b> for the model.
 * It contains access methods for the meta objects to access via {@link org.eclipse.emf.ecore.EPackage.EClassifier#getEPackage()}.
 * <!-- end-user-doc -->
 * @generated
 */
public interface PersonsPackage extends EPackage {
	/**
	 * The package namespace URI.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_URI = "http://persons";

	/**
	 * The package namespace name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNAME = "Persons";

	/**
	 * The singleton instance of the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	PersonsPackage eINSTANCE = Persons.impl.PersonsPackageImpl.init();

	/**
	 * Returns the meta object for class '{@link Persons.PersonRegister <em>Person Register</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Person Register</em>'.
	 * @model
	 * @generated
	 */
	EClass getPersonRegister();

	/**
	 * Returns the meta object for the attribute '{@link Persons.PersonRegister#getPersons <em>Persons</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Persons</em>'.
	 * @model opposite="personsInverse" containment="true"
	 * @generated
	 */
	EReference getPersonRegister_Persons();

	/**
	 * Returns the meta object for class '{@link Persons.Person <em>Person</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Person</em>'.
	 * @model abstract="true"
	 * @generated
	 */
	EClass getPerson();

	/**
	 * Returns the meta object for the attribute '{@link Persons.Person#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @model
	 * @generated
	 */
	EAttribute getPerson_Name();

	/**
	 * Returns the meta object for the attribute '{@link Persons.Person#getBirthday <em>Birthday</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Birthday</em>'.
	 * @model default="0000-1-1"
	 * @generated
	 */
	EAttribute getPerson_Birthday();

	/**
	 * Returns the meta object for the container reference '{@link Persons.Person#getPersonsInverse <em>Persons Inverse</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Persons Inverse</em>'.
	 * @model opposite="persons" transient="false"
	 * @generated
	 */
	EReference getPerson_PersonsInverse();

	/**
	 * Returns the meta object for class '{@link Persons.Male <em>Male</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Male</em>'.
	 * @model
	 * @generated
	 */
	EClass getMale();

	/**
	 * Returns the meta object for class '{@link Persons.Female <em>Female</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Female</em>'.
	 * @model
	 * @generated
	 */
	EClass getFemale();

} // PersonsPackage