package Transformation;

import java.util.Calendar;
import java.util.Date;

import Families.FamilyRegister;
import Families.Family;
import Families.FamilyMember;
import Families.FamiliesFactory;

import Persons.PersonRegister;
import Persons.Person;
import Persons.Male;
import Persons.Female;
import Persons.PersonsFactory;

/**
 * <!-- begin-user-doc -->
 * Transforms a {@link FamilyRegister} containing {@link Family} objects with {@link FamilyMember}s
 * into a {@link PersonRegister} containing {@link Person} objects (Male or Female).
 * 
 * This is the forward transformation from the Families model to the Persons model.
 * 
 * <p>
 * The transformation:
 * <ul>
 *   <li>Iterates through all Families in the FamilyRegister</li>
 *   <li>Transforms each father and son to a Male Person</li>
 *   <li>Transforms each mother and daughter to a Female Person</li>
 *   <li>Sets all birthdays to the default value (0000-1-1)</li>
 *   <li>Establishes bidirectional links between Person and PersonRegister</li>
 * </ul>
 * </p>
 * 
 * @see FamiliesToPersonsBidirectionalTransformation
 * @generated
 */
public class FamiliesToPersonsForwardTransformer {
	
	/**
	 * The default birthday date used when transforming FamilyMembers to Persons.
	 * This is the year, month, day as defined in the Persons model.
	 * @generated
	 */
	public static final Date DEFAULT_BIRTHDAY;
	
	static {
		Calendar cal = Calendar.getInstance();
		cal.set(Calendar.YEAR, 0);
		cal.set(Calendar.MONTH, 0); // January (0-based)
		cal.set(Calendar.DAY_OF_MONTH, 1);
		cal.set(Calendar.HOUR_OF_DAY, 0);
		cal.set(Calendar.MINUTE, 0);
		cal.set(Calendar.SECOND, 0);
		cal.set(Calendar.MILLISECOND, 0);
		DEFAULT_BIRTHDAY = cal.getTime();
	}

	/**
	 * Transforms the given {@link FamilyRegister} into a {@link PersonRegister}.
	 * 
	 * @param familyRegister the source FamilyRegister to transform
	 * @return a new PersonRegister containing all transformed persons
	 * @throws IllegalArgumentException if familyRegister is null
	 * @generated
	 */
	public PersonRegister transform(FamilyRegister familyRegister) {
		if (familyRegister == null) {
			throw new IllegalArgumentException("FamilyRegister cannot be null");
		}
		
		// Step 1.1: Create PersonRegister from FamilyRegister
		PersonsFactory personsFactory = PersonsFactory.eINSTANCE;
		PersonRegister personRegister = personsFactory.createPersonRegister();
		
		// Step 1.2: Iterate through all Families
		for (Family family : familyRegister.getFamilies()) {
			// Step 1.3: Transform Male FamilyMembers (father and sons)
			transformMaleFamilyMember(family.getFather(), personRegister);
			for (FamilyMember son : family.getSons()) {
				transformMaleFamilyMember(son, personRegister);
			}
			
			// Step 1.4: Transform Female FamilyMembers (mother and daughters)
			transformFemaleFamilyMember(family.getMother(), personRegister);
			for (FamilyMember daughter : family.getDaughters()) {
				transformFemaleFamilyMember(daughter, personRegister);
			}
		}
		
		// Step 1.5: Complete Forward Transformation
		return personRegister;
	}

	/**
	 * Transforms a male FamilyMember (father or son) to a Male Person.
	 * 
	 * @param familyMember the source FamilyMember, may be null
	 * @param personRegister the target PersonRegister to add the transformed person to
	 * @generated
	 */
	private void transformMaleFamilyMember(FamilyMember familyMember, PersonRegister personRegister) {
		if (familyMember == null) {
			return;
		}
		
		// Create a new Male instance using PersonsFactory
		PersonsFactory personsFactory = PersonsFactory.eINSTANCE;
		Male male = personsFactory.createMale();
		
		// Copy the name attribute from FamilyMember to Person
		male.setName(familyMember.getName());
		
		// Set birthday to default value
		male.setBirthday(DEFAULT_BIRTHDAY);
		
		// Add to PersonRegister's persons list
		personRegister.getPersons().add(male);
		
		// Establish bidirectional link (set personsInverse reference)
		male.setPersonsInverse(personRegister);
	}

	/**
	 * Transforms a female FamilyMember (mother or daughter) to a Female Person.
	 * 
	 * @param familyMember the source FamilyMember, may be null
	 * @param personRegister the target PersonRegister to add the transformed person to
	 * @generated
	 */
	private void transformFemaleFamilyMember(FamilyMember familyMember, PersonRegister personRegister) {
		if (familyMember == null) {
			return;
		}
		
		// Create a new Female instance using PersonsFactory
		PersonsFactory personsFactory = PersonsFactory.eINSTANCE;
		Female female = personsFactory.createFemale();
		
		// Copy the name attribute from FamilyMember to Person
		female.setName(familyMember.getName());
		
		// Set birthday to default value
		female.setBirthday(DEFAULT_BIRTHDAY);
		
		// Add to PersonRegister's persons list
		personRegister.getPersons().add(female);
		
		// Establish bidirectional link (set personsInverse reference)
		female.setPersonsInverse(personRegister);
	}

} // FamiliesToPersonsForwardTransformer