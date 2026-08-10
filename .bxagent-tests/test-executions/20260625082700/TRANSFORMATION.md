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
Bidirectional transformation is required:

1. **Forward Transformation (Families → Persons)**: 
   - Transform FamilyRegister containing Families into PersonRegister containing Persons
   - Each FamilyMember becomes a Person (Male or Female based on their role)
   - Family structure information is flattened; persons are collected in a single register
   - Family name information may need to be preserved or discarded

2. **Backward Transformation (Persons → Families)**:
   - Transform PersonRegister back into FamilyRegister with Families
   - Need to reconstruct family structure from flat person list
   - Requires heuristics or additional metadata to determine family groupings
   - Gender information determines possible roles (Male can be father/son, Female can be mother/daughter)
--- END TRANSFORMATION DIRECTION ---

---

## 3. Identified Difficulties

Several difficulties with the transformation itself have been identified:
--- BEGIN DIFFICULTIES ---
## Difficulty 1: Structural Mismatch Between Hierarchical and Flat Models

**Challenge**: The Families model uses a nested structure (FamilyRegister contains multiple Families, each Family contains FamilyMembers in specific roles), while the Persons model is flat (PersonRegister contains a simple list of Persons). 

- **Forward direction**: Straightforward - flatten all FamilyMembers from all Families into a single person list
- **Backward direction**: Highly ambiguous - need to reconstruct family groupings from a flat list without explicit family membership information

**Configuration needed**: Strategy for reconstructing families in backward transformation (e.g., create single family, group by name patterns, or require external configuration)

## Difficulty 2: Gender vs Role Mapping

**Challenge**: Families model encodes gender implicitly through role positions (father/mother/sons/daughters), while Persons model has explicit Male/Female subclasses.

- **Forward mapping**: Clear - father/sons → Male, mother/daughters → Female
- **Backward mapping**: Lossy - knowing someone is Male doesn't tell us if they should be a father or son in a family

**Impact**: Round-trip transformation cannot perfectly restore original family structure

## Difficulty 3: Missing Birthday Information in Source

**Challenge**: Person has a `birthday` attribute (with default "0000-1-1") that doesn't exist in FamilyMember.

- **Forward direction**: Must set default birthday value for all transformed persons
- **Backward direction**: Preserve existing birthday values, no special handling needed

## Difficulty 4: Family Name Preservation

**Challenge**: Family has a `name` attribute that has no direct counterpart in the Persons model.

- **Forward direction**: Family name information will be lost unless encoded in person names or stored as metadata
- **Backward direction**: Cannot restore original family names without additional information

## Difficulty 5: Bidirectional Consistency

**Challenge**: Due to information loss in flattening (family memberships, family names, specific roles), perfect round-trip consistency is impossible.

**Strategy**: Define acceptable consistency guarantees - e.g., all persons preserved with correct gender and names, but family structure may differ after round-trip
--- END DIFFICULTIES ---

Please note that you have to provide Configuration parameters if there are multiple strategies to resolve a difficulty.

---

## 4. Implementation Steps

--- BEGIN IMPLEMENTATION STEPS ---
## Implementation Steps

### Phase 1: Forward Transformation (Families → Persons)

#### Step 1.1: Create PersonRegister from FamilyRegister
- Instantiate a new PersonRegister using PersonsFactory
- This will serve as the container for all transformed persons

#### Step 1.2: Iterate Through All Families
- For each Family in the source FamilyRegister's families list:
  - Access father, mother, sons, and daughters collections
  - Note: family name may be logged/ignored as it has no target equivalent

#### Step 1.3: Transform Male FamilyMembers
- For each father (if present):
  - Create a new Male instance using PersonsFactory
  - Copy the name attribute from FamilyMember to Person
  - Set birthday to default value ("0000-1-1" or equivalent Date object)
  - Add to PersonRegister's persons list
  - Establish bidirectional link (set personsInverse reference)
  
- For each son in the sons list:
  - Create a new Male instance using PersonsFactory
  - Copy the name attribute
  - Set birthday to default value
  - Add to PersonRegister's persons list
  - Establish bidirectional link

#### Step 1.4: Transform Female FamilyMembers
- For each mother (if present):
  - Create a new Female instance using PersonsFactory
  - Copy the name attribute from FamilyMember to Person
  - Set birthday to default value
  - Add to PersonRegister's persons list
  - Establish bidirectional link
  
- For each daughter in the daughters list:
  - Create a new Female instance using PersonsFactory
  - Copy the name attribute
  - Set birthday to default value
  - Add to PersonRegister's persons list
  - Establish bidirectional link

#### Step 1.5: Complete Forward Transformation
- Ensure all EMF containment relationships are properly established
- Return the populated PersonRegister

---

### Phase 2: Backward Transformation (Persons → Families)

#### Step 2.1: Create FamilyRegister from PersonRegister
- Instantiate a new FamilyRegister using FamiliesFactory
- This will serve as the container for reconstructed families

#### Step 2.2: Strategy Selection for Family Reconstruction
- **Configuration Option A**: Create a single family containing all persons
  - Assign first male as father, first female as mother, remaining as children
  - Simple but may not reflect intended structure
  
- **Configuration Option B**: Preserve existing family if metadata available
  - Check for any preserved family identification in person names or extended attributes
  - Group persons accordingly
  
- **Configuration Option C**: Create individual family units
  - Each person becomes their own "family" or no family structure is created
  - Conservative approach that doesn't make assumptions

#### Step 2.3: Separate Persons by Gender
- Iterate through all persons in the PersonRegister
- Collect Male instances into one list
- Collect Female instances into another list
- Maintain original order if possible for deterministic behavior

#### Step 2.4: Reconstruct Family Structure (Based on Selected Strategy)
- Using the chosen strategy from Step 2.2:
  - Create Family instance(s) using FamiliesFactory
  - Assign males to father/sons roles based on strategy
  - Assign females to mother/daughters roles based on strategy
  - Set family names to default or derive from person names if strategy allows
  - Add families to FamilyRegister

#### Step 2.5: Establish Bidirectional Links
- For each FamilyMember created:
  - Set appropriate inverse references (fatherInverse, motherInverse, sonsInverse, daughtersInverse)
  - Ensure EMF containment is properly maintained
  - Verify that getFatherInverse(), getMotherInverse(), etc. return correct values

#### Step 2.6: Complete Backward Transformation
- Ensure all EMF containment relationships are properly established
- Return the populated FamilyRegister

---

### Phase 3: Validation and Testing

#### Step 3.1: Implement Forward Transformation Tests
- Test with empty FamilyRegister
- Test with single family containing all member types
- Test with multiple families
- Test with families missing certain members (no father, no children, etc.)
- Verify all persons are created with correct gender subclasses
- Verify names are preserved correctly
- Verify birthday defaults are set

#### Step 3.2: Implement Backward Transformation Tests
- Test with empty PersonRegister
- Test with only males
- Test with only females
- Test with mixed genders
- Verify family structure is created according to selected strategy
- Verify bidirectional links are established

#### Step 3.3: Implement Round-Trip Consistency Tests
- Transform Families → Persons → Families
- Compare final Families model with original
- Document acceptable differences (expected information loss)
- Verify critical properties are preserved (all persons exist with correct names and genders)

#### Step 3.4: Edge Case Handling
- Handle null/missing references gracefully
- Ensure proper cleanup of intermediate objects if transformation fails
- Validate input models before transformation begins
--- END IMPLEMENTATION STEPS ---