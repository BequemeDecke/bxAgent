/**
 */
package Families;

import org.eclipse.emf.ecore.EPackage;

/**
 * <!-- begin-user-doc -->
 * The <b>Package</b> for the model.
 * It contains access methods for the meta objects to access via {@link org.eclipse.emf.ecore.EPackage.EClassifier#getEPackage()}.
 * <!-- end-user-doc -->
 * @generated
 */
public interface FamiliesPackage extends EPackage {
	/**
	 * The package namespace URI.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_URI = "http://families";

	/**
	 * The package namespace name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNAME = "Families";

	/**
	 * The singleton instance of the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	FamiliesPackage eINSTANCE = Families.impl.FamiliesPackageImpl.init();

	/**
	 * Returns the meta object for class '{@link Families.FamilyRegister <em>Family Register</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Family Register</em>'.
	 * @model
	 * @generated
	 */
	EClass getFamilyRegister();

	/**
	 * Returns the meta object for the attribute '{@link Families.FamilyRegister#getFamilies <em>Families</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Families</em>'.
	 * @model opposite="familiesInverse" containment="true"
	 * @generated
	 */
	EReference getFamilyRegister_Families();

	/**
	 * Returns the meta object for class '{@link Families.Family <em>Family</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Family</em>'.
	 * @model
	 * @generated
	 */
	EClass getFamily();

	/**
	 * Returns the meta object for the containment reference '{@link Families.Family#getFather <em>Father</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Father</em>'.
	 * @model opposite="fatherInverse" containment="true"
	 * @generated
	 */
	EReference getFamily_Father();

	/**
	 * Returns the meta object for the containment reference '{@link Families.Family#getMother <em>Mother</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Mother</em>'.
	 * @model opposite="motherInverse" containment="true"
	 * @generated
	 */
	EReference getFamily_Mother();

	/**
	 * Returns the meta object for the containment reference list '{@link Families.Family#getSons <em>Sons</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Sons</em>'.
	 * @model opposite="sonsInverse" containment="true"
	 * @generated
	 */
	EReference getFamily_Sons();

	/**
	 * Returns the meta object for the containment reference list '{@link Families.Family#getDaughters <em>Daughters</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Daughters</em>'.
	 * @model opposite="daughtersInverse" containment="true"
	 * @generated
	 */
	EReference getFamily_Daughters();

	/**
	 * Returns the meta object for the attribute '{@link Families.Family#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @model
	 * @generated
	 */
	EAttribute getFamily_Name();

	/**
	 * Returns the meta object for the container reference '{@link Families.Family#getFamiliesInverse <em>Families Inverse</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Families Inverse</em>'.
	 * @model opposite="families" transient="false"
	 * @generated
	 */
	EReference getFamily_FamiliesInverse();

	/**
	 * Returns the meta object for class '{@link Families.FamilyMember <em>Family Member</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Family Member</em>'.
	 * @model
	 * @generated
	 */
	EClass getFamilyMember();

	/**
	 * Returns the meta object for the attribute '{@link Families.FamilyMember#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @model
	 * @generated
	 */
	EAttribute getFamilyMember_Name();

	/**
	 * Returns the meta object for the container reference '{@link Families.FamilyMember#getFatherInverse <em>Father Inverse</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Father Inverse</em>'.
	 * @model opposite="father" transient="false"
	 * @generated
	 */
	EReference getFamilyMember_FatherInverse();

	/**
	 * Returns the meta object for the container reference '{@link Families.FamilyMember#getMotherInverse <em>Mother Inverse</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Mother Inverse</em>'.
	 * @model opposite="mother" transient="false"
	 * @generated
	 */
	EReference getFamilyMember_MotherInverse();

	/**
	 * Returns the meta object for the container reference '{@link Families.FamilyMember#getSonsInverse <em>Sons Inverse</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Sons Inverse</em>'.
	 * @model opposite="sons" transient="false"
	 * @generated
	 */
	EReference getFamilyMember_SonsInverse();

	/**
	 * Returns the meta object for the container reference '{@link Families.FamilyMember#getDaughtersInverse <em>Daughters Inverse</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Daughters Inverse</em>'.
	 * @model opposite="daughters" transient="false"
	 * @generated
	 */
	EReference getFamilyMember_DaughtersInverse();

} // FamiliesPackage