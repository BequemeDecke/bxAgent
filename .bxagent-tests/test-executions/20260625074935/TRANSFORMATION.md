---
source_model_package: Families
target_model_package: Persons
iteration: 1
---

# Transformation Plan

## 1. Model Implementations

The implementation of the source model looks as followed:
--- BEGIN SOURCE MODEL ---
.bxagent-tests/setup-files/Families/FamilyMember.java:
/**
 */
package Families;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Family Member</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link Families.FamilyMember#getName <em>Name</em>}</li>
 *   <li>{@link Families.FamilyMember#getFatherInverse <em>Father Inverse</em>}</li>
 *   <li>{@link Families.FamilyMember#getMotherInverse <em>Mother Inverse</em>}</li>
 *   <li>{@link Families.FamilyMember#getSonsInverse <em>Sons Inverse</em>}</li>
 *   <li>{@link Families.FamilyMember#getDaughtersInverse <em>Daughters Inverse</em>}</li>
 * </ul>
 *
 * @see Families.FamiliesPackage#getFamilyMember()
 * @model
 * @generated
 */
public interface FamilyMember extends EObject {
	/**
	 * Returns the value of the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Name</em>' attribute.
	 * @see #setName(String)
	 * @see Families.FamiliesPackage#getFamilyMember_Name()
	 * @model
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link Families.FamilyMember#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Father Inverse</b></em>' container reference.
	 * It is bidirectional and its opposite is '{@link Families.Family#getFather <em>Father</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Father Inverse</em>' container reference.
	 * @see #setFatherInverse(Family)
	 * @see Families.FamiliesPackage#getFamilyMember_FatherInverse()
	 * @see Families.Family#getFather
	 * @model opposite="father" transient="false"
	 * @generated
	 */
	Family getFatherInverse();

	/**
	 * Sets the value of the '{@link Families.FamilyMember#getFatherInverse <em>Father Inverse</em>}' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Father Inverse</em>' container reference.
	 * @see #getFatherInverse()
	 * @generated
	 */
	void setFatherInverse(Family value);

	/**
	 * Returns the value of the '<em><b>Mother Inverse</b></em>' container reference.
	 * It is bidirectional and its opposite is '{@link Families.Family#getMother <em>Mother</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Mother Inverse</em>' container reference.
	 * @see #setMotherInverse(Family)
	 * @see Families.FamiliesPackage#getFamilyMember_MotherInverse()
	 * @see Families.Family#getMother
	 * @model opposite="mother" transient="false"
	 * @generated
	 */
	Family getMotherInverse();

	/**
	 * Sets the value of the '{@link Families.FamilyMember#getMotherInverse <em>Mother Inverse</em>}' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Mother Inverse</em>' container reference.
	 * @see #getMotherInverse()
	 * @generated
	 */
	void setMotherInverse(Family value);

	/**
	 * Returns the value of the '<em><b>Sons Inverse</b></em>' container reference.
	 * It is bidirectional and its opposite is '{@link Families.Family#getSons <em>Sons</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Sons Inverse</em>' container reference.
	 * @see #setSonsInverse(Family)
	 * @see Families.FamiliesPackage#getFamilyMember_SonsInverse()
	 * @see Families.Family#getSons
	 * @model opposite="sons" transient="false"
	 * @generated
	 */
	Family getSonsInverse();

	/**
	 * Sets the value of the '{@link Families.FamilyMember#getSonsInverse <em>Sons Inverse</em>}' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Sons Inverse</em>' container reference.
	 * @see #getSonsInverse()
	 * @generated
	 */
	void setSonsInverse(Family value);

	/**
	 * Returns the value of the '<em><b>Daughters Inverse</b></em>' container reference.
	 * It is bidirectional and its opposite is '{@link Families.Family#getDaughters <em>Daughters</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Daughters Inverse</em>' container reference.
	 * @see #setDaughtersInverse(Family)
	 * @see Families.FamiliesPackage#getFamilyMember_DaughtersInverse()
	 * @see Families.Family#getDaughters
	 * @model opposite="daughters" transient="false"
	 * @generated
	 */
	Family getDaughtersInverse();

	/**
	 * Sets the value of the '{@link Families.FamilyMember#getDaughtersInverse <em>Daughters Inverse</em>}' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Daughters Inverse</em>' container reference.
	 * @see #getDaughtersInverse()
	 * @generated
	 */
	void setDaughtersInverse(Family value);

} // FamilyMember


.bxagent-tests/setup-files/Families/FamiliesFactory.java:
/**
 */
package Families;

import org.eclipse.emf.ecore.EFactory;

/**
 * <!-- begin-user-doc -->
 * The <b>Factory</b> for the model.
 * It provides a create method for each non-abstract class of the model.
 * <!-- end-user-doc -->
 * @see Families.FamiliesPackage
 * @generated
 */
public interface FamiliesFactory extends EFactory {
	/**
	 * The singleton instance of the factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	FamiliesFactory eINSTANCE = Families.impl.FamiliesFactoryImpl.init();

	/**
	 * Returns a new object of class '<em>Family Register</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Family Register</em>'.
	 * @generated
	 */
	FamilyRegister createFamilyRegister();

	/**
	 * Returns a new object of class '<em>Family</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Family</em>'.
	 * @generated
	 */
	Family createFamily();

	/**
	 * Returns a new object of class '<em>Family Member</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Family Member</em>'.
	 * @generated
	 */
	FamilyMember createFamilyMember();

	/**
	 * Returns the package supported by this factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the package supported by this factory.
	 * @generated
	 */
	FamiliesPackage getFamiliesPackage();

} //FamiliesFactory


.bxagent-tests/setup-files/Families/FamilyRegister.java:
/**
 */
package Families;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Family Register</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link Families.FamilyRegister#getFamilies <em>Families</em>}</li>
 * </ul>
 *
 * @see Families.FamiliesPackage#getFamilyRegister()
 * @model
 * @generated
 */
public interface FamilyRegister extends EObject {
	/**
	 * Returns the value of the '<em><b>Families</b></em>' containment reference list.
	 * The list contents are of type {@link Families.Family}.
	 * It is bidirectional and its opposite is '{@link Families.Family#getFamiliesInverse <em>Families Inverse</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Families</em>' containment reference list.
	 * @see Families.FamiliesPackage#getFamilyRegister_Families()
	 * @see Families.Family#getFamiliesInverse
	 * @model opposite="familiesInverse" containment="true"
	 * @generated
	 */
	EList<Family> getFamilies();

} // FamilyRegister


.bxagent-tests/setup-files/Families/Family.java:
/**
 */
package Families;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Family</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link Families.Family#getFather <em>Father</em>}</li>
 *   <li>{@link Families.Family#getMother <em>Mother</em>}</li>
 *   <li>{@link Families.Family#getSons <em>Sons</em>}</li>
 *   <li>{@link Families.Family#getDaughters <em>Daughters</em>}</li>
 *   <li>{@link Families.Family#getName <em>Name</em>}</li>
 *   <li>{@link Families.Family#getFamiliesInverse <em>Families Inverse</em>}</li>
 * </ul>
 *
 * @see Families.FamiliesPackage#getFamily()
 * @model
 * @generated
 */
public interface Family extends EObject {
	/**
	 * Returns the value of the '<em><b>Father</b></em>' containment reference.
	 * It is bidirectional and its opposite is '{@link Families.FamilyMember#getFatherInverse <em>Father Inverse</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Father</em>' containment reference.
	 * @see #setFather(FamilyMember)
	 * @see Families.FamiliesPackage#getFamily_Father()
	 * @see Families.FamilyMember#getFatherInverse
	 * @model opposite="fatherInverse" containment="true"
	 * @generated
	 */
	FamilyMember getFather();

	/**
	 * Sets the value of the '{@link Families.Family#getFather <em>Father</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Father</em>' containment reference.
	 * @see #getFather()
	 * @generated
	 */
	void setFather(FamilyMember value);

	/**
	 * Returns the value of the '<em><b>Mother</b></em>' containment reference.
	 * It is bidirectional and its opposite is '{@link Families.FamilyMember#getMotherInverse <em>Mother Inverse</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Mother</em>' containment reference.
	 * @see #setMother(FamilyMember)
	 * @see Families.FamiliesPackage#getFamily_Mother()
	 * @see Families.FamilyMember#getMotherInverse
	 * @model opposite="motherInverse" containment="true"
	 * @generated
	 */
	FamilyMember getMother();

	/**
	 * Sets the value of the '{@link Families.Family#getMother <em>Mother</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Mother</em>' containment reference.
	 * @see #getMother()
	 * @generated
	 */
	void setMother(FamilyMember value);

	/**
	 * Returns the value of the '<em><b>Sons</b></em>' containment reference list.
	 * The list contents are of type {@link Families.FamilyMember}.
	 * It is bidirectional and its opposite is '{@link Families.FamilyMember#getSonsInverse <em>Sons Inverse</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Sons</em>' containment reference list.
	 * @see Families.FamiliesPackage#getFamily_Sons()
	 * @see Families.FamilyMember#getSonsInverse
	 * @model opposite="sonsInverse" containment="true"
	 * @generated
	 */
	EList<FamilyMember> getSons();

	/**
	 * Returns the value of the '<em><b>Daughters</b></em>' containment reference list.
	 * The list contents are of type {@link Families.FamilyMember}.
	 * It is bidirectional and its opposite is '{@link Families.FamilyMember#getDaughtersInverse <em>Daughters Inverse</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Daughters</em>' containment reference list.
	 * @see Families.FamiliesPackage#getFamily_Daughters()
	 * @see Families.FamilyMember#getDaughtersInverse
	 * @model opposite="daughtersInverse" containment="true"
	 * @generated
	 */
	EList<FamilyMember> getDaughters();

	/**
	 * Returns the value of the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Name</em>' attribute.
	 * @see #setName(String)
	 * @see Families.FamiliesPackage#getFamily_Name()
	 * @model
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link Families.Family#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Families Inverse</b></em>' container reference.
	 * It is bidirectional and its opposite is '{@link Families.FamilyRegister#getFamilies <em>Families</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Families Inverse</em>' container reference.
	 * @see #setFamiliesInverse(FamilyRegister)
	 * @see Families.FamiliesPackage#getFamily_FamiliesInverse()
	 * @see Families.FamilyRegister#getFamilies
	 * @model opposite="families" transient="false"
	 * @generated
	 */
	FamilyRegister getFamiliesInverse();

	/**
	 * Sets the value of the '{@link Families.Family#getFamiliesInverse <em>Families Inverse</em>}' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Families Inverse</em>' container reference.
	 * @see #getFamiliesInverse()
	 * @generated
	 */
	void setFamiliesInverse(FamilyRegister value);

} // Family

--- END SOURCE MODEL ---

The implementation of the target model looks as followed:
--- BEGIN TARGET MODEL ---
.bxagent-tests/setup-files/Persons/PersonRegister.java:
/**
 */
package Persons;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Person Register</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link Persons.PersonRegister#getPersons <em>Persons</em>}</li>
 * </ul>
 *
 * @see Persons.PersonsPackage#getPersonRegister()
 * @model
 * @generated
 */
public interface PersonRegister extends EObject {
	/**
	 * Returns the value of the '<em><b>Persons</b></em>' containment reference list.
	 * The list contents are of type {@link Persons.Person}.
	 * It is bidirectional and its opposite is '{@link Persons.Person#getPersonsInverse <em>Persons Inverse</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Persons</em>' containment reference list.
	 * @see Persons.PersonsPackage#getPersonRegister_Persons()
	 * @see Persons.Person#getPersonsInverse
	 * @model opposite="personsInverse" containment="true"
	 * @generated
	 */
	EList<Person> getPersons();

} // PersonRegister


.bxagent-tests/setup-files/Persons/Person.java:
/**
 */
package Persons;

import java.util.Date;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Person</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link Persons.Person#getName <em>Name</em>}</li>
 *   <li>{@link Persons.Person#getBirthday <em>Birthday</em>}</li>
 *   <li>{@link Persons.Person#getPersonsInverse <em>Persons Inverse</em>}</li>
 * </ul>
 *
 * @see Persons.PersonsPackage#getPerson()
 * @model abstract="true"
 * @generated
 */
public interface Person extends EObject {
	/**
	 * Returns the value of the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Name</em>' attribute.
	 * @see #setName(String)
	 * @see Persons.PersonsPackage#getPerson_Name()
	 * @model
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link Persons.Person#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Birthday</b></em>' attribute.
	 * The default value is <code>"0000-1-1"</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Birthday</em>' attribute.
	 * @see #setBirthday(Date)
	 * @see Persons.PersonsPackage#getPerson_Birthday()
	 * @model default="0000-1-1"
	 * @generated
	 */
	Date getBirthday();

	/**
	 * Sets the value of the '{@link Persons.Person#getBirthday <em>Birthday</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Birthday</em>' attribute.
	 * @see #getBirthday()
	 * @generated
	 */
	void setBirthday(Date value);

	/**
	 * Returns the value of the '<em><b>Persons Inverse</b></em>' container reference.
	 * It is bidirectional and its opposite is '{@link Persons.PersonRegister#getPersons <em>Persons</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Persons Inverse</em>' container reference.
	 * @see #setPersonsInverse(PersonRegister)
	 * @see Persons.PersonsPackage#getPerson_PersonsInverse()
	 * @see Persons.PersonRegister#getPersons
	 * @model opposite="persons" transient="false"
	 * @generated
	 */
	PersonRegister getPersonsInverse();

	/**
	 * Sets the value of the '{@link Persons.Person#getPersonsInverse <em>Persons Inverse</em>}' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Persons Inverse</em>' container reference.
	 * @see #getPersonsInverse()
	 * @generated
	 */
	void setPersonsInverse(PersonRegister value);

} // Person


.bxagent-tests/setup-files/Persons/PersonsFactory.java:
/**
 */
package Persons;

import org.eclipse.emf.ecore.EFactory;

/**
 * <!-- begin-user-doc -->
 * The <b>Factory</b> for the model.
 * It provides a create method for each non-abstract class of the model.
 * <!-- end-user-doc -->
 * @see Persons.PersonsPackage
 * @generated
 */
public interface PersonsFactory extends EFactory {
	/**
	 * The singleton instance of the factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	PersonsFactory eINSTANCE = Persons.impl.PersonsFactoryImpl.init();

	/**
	 * Returns a new object of class '<em>Person Register</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Person Register</em>'.
	 * @generated
	 */
	PersonRegister createPersonRegister();

	/**
	 * Returns a new object of class '<em>Male</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Male</em>'.
	 * @generated
	 */
	Male createMale();

	/**
	 * Returns a new object of class '<em>Female</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Female</em>'.
	 * @generated
	 */
	Female createFemale();

	/**
	 * Returns the package supported by this factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the package supported by this factory.
	 * @generated
	 */
	PersonsPackage getPersonsPackage();

} //PersonsFactory

--- END TARGET MODEL ---

## 2. Transformation Direction

Bidirectional transformation is required, meaning source to target and target to source:
--- BEGIN TRANSFORMATION DIRECTION ---
Bidirectional transformation required:
1. Source to Target (Families → Persons): Transform FamilyRegister with Family structures into PersonRegister with flat Person list. Each FamilyMember becomes a Person (Male or Female based on their role in family). Preserve name information. Birthday will need default value since not present in source.
2. Target to Source (Persons → Families): Transform PersonRegister into FamilyRegister. Need to reconstruct Family structures from flat Person list. This requires determining family groupings and roles (father, mother, sons, daughters) which may not be explicitly encoded in the target model.
--- END TRANSFORMATION DIRECTION ---

---

## 3. Identified Difficulties

Several difficulties with the transformation itself have been identified:
--- BEGIN DIFFICULTIES ---
## Difficulty 1: Gender Determination in Source-to-Target Transformation
**Challenge**: In the Families model, gender is implicit through roles (Father, Mother, Sons, Daughters). In the Persons model, gender is explicit through Male/Female subclasses.
**Resolution Strategy**: Map FamilyMember roles to Person subclasses:
- Father → Male
- Mother → Female  
- Sons → Male
- Daughters → Female
This is straightforward as the role clearly indicates gender.

## Difficulty 2: Family Structure Reconstruction in Target-to-Source Transformation
**Challenge**: The Persons model is a flat list of Person objects without explicit family relationship information. Reconstructing Family structures (grouping persons into families and assigning roles as father/mother/sons/daughters) is ambiguous without additional metadata.
**Resolution Strategy**: This requires external configuration or heuristics. Possible approaches:
- Use naming conventions or person attributes to infer family groupings
- Require additional metadata in Person objects (not currently in model)
- Create singleton families or use a default grouping strategy
**Configuration Needed**: A strategy parameter to determine how to group persons into families during reverse transformation.

## Difficulty 3: Missing Birthday Information
**Challenge**: The target Person model includes a Birthday attribute (with default "0000-1-1") that does not exist in the source FamilyMember model.
**Resolution Strategy**: 
- Forward transformation (Families→Persons): Use the default birthday value "0000-1-1" for all created Person objects
- Reverse transformation (Persons→Families): Birthday information is simply not transferred back as it doesn't exist in the target model

## Difficulty 4: Bidirectional Consistency and Information Loss
**Challenge**: Transforming Families→Persons loses family structure information (which persons belong to which family and their roles within that family). A round-trip transformation may not preserve the original structure.
**Resolution Strategy**: Accept that perfect round-trip consistency cannot be achieved without additional metadata. Document this limitation and define expected behavior:
- Families→Persons→Families may result in different family groupings
- Consider adding optional metadata fields if round-trip consistency is critical
--- END DIFFICULTIES ---

Please note that you have to provide Configuration parameters if there are multiple strategies to resolve a difficulty.

---

## 4. Implementation Steps

--- BEGIN IMPLEMENTATION STEPS ---
## Implementation Steps

### Phase 1: Source to Target Transformation (Families → Persons)

**Step 1.1: Create PersonRegister from FamilyRegister**
- Instantiate a new PersonRegister object using PersonsFactory
- This will be the root container for all transformed Person objects

**Step 1.2: Iterate through all Families in the source FamilyRegister**
- Access the list of Family objects from the source FamilyRegister.getFamilies()
- For each Family, process all contained FamilyMember objects

**Step 1.3: Transform each FamilyMember to a Person**
For each FamilyMember in each Family:
- Determine the gender based on the member's role:
  - If the member is referenced as Father or in Sons list → create Male instance
  - If the member is referenced as Mother or in Daughters list → create Female instance
- Copy the Name attribute from FamilyMember to Person
- Set Birthday to the default value "0000-1-1" (Date representation)
- Add the created Person to the PersonRegister's persons list
- Establish the bidirectional link by setting Person's personsInverse reference

**Step 1.4: Handle duplicate FamilyMembers**
- Track already-transformed FamilyMembers (using object identity or a mapping) to avoid creating duplicate Person objects if a FamilyMember appears in multiple contexts
- Note: In the current model structure, each FamilyMember should belong to exactly one Family due to containment relationships

### Phase 2: Target to Source Transformation (Persons → Families)

**Step 2.1: Create FamilyRegister from PersonRegister**
- Instantiate a new FamilyRegister object using FamiliesFactory
- This will be the root container for reconstructed Family structures

**Step 2.2: Group Persons into Families**
- Apply a grouping strategy to partition the flat list of Persons into family units
- Default strategy: Create one Family per Person (singleton families) OR group all persons into a single Family
- Alternative strategies may use naming patterns or external configuration

**Step 2.3: Reconstruct Family structure for each group**
For each group of Persons designated as a Family:
- Create a new Family object
- Assign a name to the Family (could be derived from members' names or use a default)
- For each Person in the group:
  - Determine their role based on gender and any available metadata:
    - Male persons could be assigned as Father or Sons
    - Female persons could be assigned as Mother or Daughters
  - Create corresponding FamilyMember objects with the person's name
  - Set appropriate containment references (father, mother, sons, daughters)
- Add the Family to the FamilyRegister's families list
- Establish bidirectional links

**Step 2.4: Handle role assignment ambiguity**
- When multiple persons of same gender exist in a family group, define rules:
  - First male encountered → Father, remaining males → Sons
  - First female encountered → Mother, remaining females → Daughters
- Document that this heuristic may not reflect original intent

### Phase 3: Bidirectional Synchronization Support

**Step 3.1: Implement change tracking mechanism**
- Maintain mappings between source and target objects during transformation
- Enable incremental updates when models change

**Step 3.2: Define merge/conflict resolution policies**
- Specify behavior when both models are modified independently
- Determine which model takes precedence for conflicting attributes

### Phase 4: Validation and Testing

**Step 4.1: Create test cases for forward transformation**
- Test with various family structures (single parent, two parents, multiple children)
- Verify correct gender assignment and name preservation

**Step 4.2: Create test cases for reverse transformation**
- Test with different person groupings
- Verify family structure reconstruction logic

**Step 4.3: Test round-trip transformations**
- Document expected information loss scenarios
- Validate that essential data (names, genders) survives round-trip
--- END IMPLEMENTATION STEPS ---