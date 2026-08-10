/**
 */
package Persons;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Female</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link Persons.Female#getName <em>Name</em>}</li>
 *   <li>{@link Persons.Female#getBirthday <em>Birthday</em>}</li>
 *   <li>{@link Persons.Female#getPersonsInverse <em>Persons Inverse</em>}</li>
 * </ul>
 *
 * @see Persons.PersonsPackage#getFemale()
 * @model
 * @generated
 */
public interface Female extends Person {
	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @model
	 * @generated
	 */
	String getName();

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @model
	 * @generated
	 */
	void setName(String value);

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @model
	 * @generated
	 */
	java.util.Date getBirthday();

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @model
	 * @generated
	 */
	void setBirthday(java.util.Date value);

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @model
	 * @generated
	 */
	PersonRegister getPersonsInverse();

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @model
	 * @generated
	 */
	void setPersonsInverse(PersonRegister value);

} // Female