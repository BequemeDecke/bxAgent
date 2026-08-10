/**
 */
package Persons;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EPackage;

/**
 * <!-- begin-user-doc -->
 * The <b>Package</b> for the model.
 * It provides access to the model explicitly and provides the parser
 * for the model.
 * <!-- end-user-doc -->
 * @see Persons.PersonsFactory
 * @generated
 */
public interface PersonsPackage extends EPackage {
	/**
	 * The package namespace URI.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_URI = "http://Persons";

	/**
	 * The package namespace name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNAME = "Persons";

	/**
	 * Returns the person register class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the class for the '<em>Person Register</em>' model element.
	 * @generated
	 */
	EClass getPersonRegister();

	/**
	 * Returns the person class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the class for the '<em>Person</em>' model element.
	 * @generated
	 */
	EClass getPerson();

	/**
	 * Returns the male class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the class for the '<em>Male</em>' model element.
	 * @generated
	 */
	EClass getMale();

	/**
	 * Returns the female class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the class for the '<em>Female</em>' model element.
	 * @generated
	 */
	EClass getFemale();

	/**
	 * Returns the default instance of the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the package instance.
	 * @generated
	 */
	PersonsPackage init();

} //PersonsPackage