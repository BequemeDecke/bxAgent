/**
 */
package Families;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EPackage;

/**
 * <!-- begin-user-doc -->
 * The <b>Package</b> for the model.
 * It provides access to the model explicitly and provides the parser
 * for the model.
 * <!-- end-user-doc -->
 * @see Families.FamiliesFactory
 * @generated
 */
public interface FamiliesPackage extends EPackage {
	/**
	 * The package namespace URI.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_URI = "http://Families";

	/**
	 * The package namespace name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNAME = "Families";

	/**
	 * Returns the family register class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the class for the '<em>Family Register</em>' model element.
	 * @generated
	 */
	EClass getFamilyRegister();

	/**
	 * Returns the family class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the class for the '<em>Family</em>' model element.
	 * @generated
	 */
	EClass getFamily();

	/**
	 * Returns the family member class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the class for the '<em>Family Member</em>' model element.
	 * @generated
	 */
	EClass getFamilyMember();

	/**
	 * Returns the default instance of the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the package instance.
	 * @generated
	 */
	FamiliesPackage init();

} //FamiliesPackage