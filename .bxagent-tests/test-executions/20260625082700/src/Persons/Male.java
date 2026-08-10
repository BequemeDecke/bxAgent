/**
 */
package Persons;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Male</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link Persons.Male#getName <em>Name</em>}</li>
 *   <li>{@link Persons.Male#getBirthday <em>Birthday</em>}</li>
 *   <li>{@link Persons.Male#getPersonsInverse <em>Persons Inverse</em>}</li>
 * </ul>
 *
 * @see Persons.PersonsPackage#getMale()
 * @model
 * @generated
 */
public interface Male extends Person {
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

} // Male