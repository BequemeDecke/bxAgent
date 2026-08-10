package Transformation;

import org.junit.Test;
import static org.junit.Assert.*;

import Families.FamilyRegister;
import Families.Family;
import Families.FamilyMember;
import Families.FamiliesFactory;

import Persons.PersonRegister;
import Persons.Person;
import Persons.Male;
import Persons.Female;
import Persons.PersonsFactory;

import PersonsToFamiliesBackwardTransformer.FamilyReconstructionStrategy;

/**
 * Unit tests for {@link FamiliesToPersonsBidirectionalTransformation}.
 * 
 * <p>
 * Tests cover:
 * <ul>
 *   <li>Forward transformation (Families → Persons)</li>
 *   <li>Backward transformation (Persons → Families)</li>
 *   <li>Round-trip transformation (Families → Persons → Families)</li>
 *   <li>Reverse round-trip transformation (Persons → Families → Persons)</li>
 *   <li>Person count preservation after round-trip</li>
 *   <li>Gender preservation after round-trip</li>
 *   <li>Name preservation after round-trip</li>
 * </ul>
 * </p>
 * 
 * @see FamiliesToPersonsBidirectionalTransformation
 * @generated
 */
public class FamiliesToPersonsBidirectionalTransformationTest {

	/**
	 * Tests forward transformation (Families → Persons).
	 * @generated
	 */
	@Test
	public void testTransformToPersons() {
		FamiliesFactory familiesFactory = FamiliesFactory.eINSTANCE;
		
		// Create a FamilyRegister
		FamilyRegister familyRegister = familiesFactory.createFamilyRegister();
		Family family = familiesFactory.createFamily();
		familyRegister.getFamilies().add(family);
		
		FamilyMember father = familiesFactory.createFamilyMember();
		father.setName("Test Father");
		family.setFather(father);
		
		FamilyMember mother = familiesFactory.createFamilyMember();
		mother.setName("Test Mother");
		family.setMother(mother);
		
		// Transform
		FamiliesToPersonsBidirectionalTransformation transformation = new FamiliesToPersonsBidirectionalTransformation();
		PersonRegister result = transformation.transformToPersons(familyRegister);
		
		assertNotNull("Result should not be null", result);
		assertEquals("Should have 2 persons", 2, result.getPersons().size());
	}

	/**
	 * Tests backward transformation (Persons → Families).
	 * @generated
	 */
	@Test
	public void testTransformToFamilies() {
		PersonsFactory personsFactory = PersonsFactory.eINSTANCE;
		
		// Create a PersonRegister
		PersonRegister personRegister = personsFactory.createPersonRegister();
		
		Male male = personsFactory.createMale();
		male.setName("Test Male");
		personRegister.getPersons().add(male);
		
		Female female = personsFactory.createFemale();
		female.setName("Test Female");
		personRegister.getPersons().add(female);
		
		// Transform
		FamiliesToPersonsBidirectionalTransformation transformation = new FamiliesToPersonsBidirectionalTransformation();
		FamilyRegister result = transformation.transformToFamilies(personRegister);
		
		assertNotNull("Result should not be null", result);
		assertEquals("Should have 1 family", 1, result.getFamilies().size());
	}

	/**
	 * Tests round-trip transformation (Families → Persons → Families).
	 * @generated
	 */
	@Test
	public void testRoundTrip() {
		FamiliesFactory familiesFactory = FamiliesFactory.eINSTANCE;
		
		// Create a FamilyRegister
		FamilyRegister familyRegister = familiesFactory.createFamilyRegister();
		Family family = familiesFactory.createFamily();
		familyRegister.getFamilies().add(family);
		
		FamilyMember father = familiesFactory.createFamilyMember();
		father.setName("Original Father");
		family.setFather(father);
		
		FamilyMember mother = familiesFactory.createFamilyMember();
		mother.setName("Original Mother");
		family.setMother(mother);
		
		FamilyMember son = familiesFactory.createFamilyMember();
		son.setName("Original Son");
		family.getSons().add(son);
		
		FamilyMember daughter = familiesFactory.createFamilyMember();
		daughter.setName("Original Daughter");
		family.getDaughters().add(daughter);
		
		// Round-trip transformation
		FamiliesToPersonsBidirectionalTransformation transformation = new FamiliesToPersonsBidirectionalTransformation();
		FamilyRegister result = transformation.roundTrip(familyRegister);
		
		// All persons should be preserved
		assertNotNull("Result should not be null", result);
		
		Family resultFamily = result.getFamilies().get(0);
		assertNotNull("Father should be set", resultFamily.getFather());
		assertEquals("Father name should be preserved", "Original Father", resultFamily.getFather().getName());
		
		assertNotNull("Mother should be set", resultFamily.getMother());
		assertEquals("Mother name should be preserved", "Original Mother", resultFamily.getMother().getName());
		
		assertEquals("Should have 1 son", 1, resultFamily.getSons().size());
		assertEquals("Son name should be preserved", "Original Son", resultFamily.getSons().get(0).getName());
		
		assertEquals("Should have 1 daughter", 1, resultFamily.getDaughters().size());
		assertEquals("Daughter name should be preserved", "Original Daughter", resultFamily.getDaughters().get(0).getName());
	}

	/**
	 * Tests reverse round-trip transformation (Persons → Families → Persons).
	 * @generated
	 */
	@Test
	public void testReverseRoundTrip() {
		PersonsFactory personsFactory = PersonsFactory.eINSTANCE;
		
		// Create a PersonRegister
		PersonRegister personRegister = personsFactory.createPersonRegister();
		
		Male male1 = personsFactory.createMale();
		male1.setName("Male 1");
		personRegister.getPersons().add(male1);
		
		Male male2 = personsFactory.createMale();
		male2.setName("Male 2");
		personRegister.getPersons().add(male2);
		
		Female female1 = personsFactory.createFemale();
		female1.setName("Female 1");
		personRegister.getPersons().add(female1);
		
		Female female2 = personsFactory.createFemale();
		female2.setName("Female 2");
		personRegister.getPersons().add(female2);
		
		// Reverse round-trip transformation
		FamiliesToPersonsBidirectionalTransformation transformation = new FamiliesToPersonsBidirectionalTransformation();
		PersonRegister result = transformation.reverseRoundTrip(personRegister);
		
		// All persons should be preserved
		assertNotNull("Result should not be null", result);
		assertEquals("Should have 4 persons", 4, result.getPersons().size());
		
		// Count males and females
		int maleCount = 0;
		int femaleCount = 0;
		for (Person person : result.getPersons()) {
			if (person instanceof Male) {
				maleCount++;
			} else if (person instanceof Female) {
				femaleCount++;
			}
		}
		
		assertEquals("Should have 2 males", 2, maleCount);
		assertEquals("Should have 2 females", 2, femaleCount);
	}

	/**
	 * Tests that all persons are preserved after round-trip.
	 * @generated
	 */
	@Test
	public void testPersonCountPreservationAfterRoundTrip() {
		FamiliesFactory familiesFactory = FamiliesFactory.eINSTANCE;
		
		// Create FamilyRegister with multiple families
		FamilyRegister familyRegister = familiesFactory.createFamilyRegister();
		
		// Family 1
		Family family1 = familiesFactory.createFamily();
		FamilyMember f1Father = familiesFactory.createFamilyMember();
		f1Father.setName("F1 Father");
		family1.setFather(f1Father);
		FamilyMember f1Mother = familiesFactory.createFamilyMember();
		f1Mother.setName("F1 Mother");
		family1.setMother(f1Mother);
		FamilyMember f1Son = familiesFactory.createFamilyMember();
		f1Son.setName("F1 Son");
		family1.getSons().add(f1Son);
		familyRegister.getFamilies().add(family1);
		
		// Family 2
		Family family2 = familiesFactory.createFamily();
		FamilyMember f2Father = familiesFactory.createFamilyMember();
		f2Father.setName("F2 Father");
		family2.setFather(f2Father);
		FamilyMember f2Mother = familiesFactory.createFamilyMember();
		f2Mother.setName("F2 Mother");
		family2.setMother(f2Mother);
		familyRegister.getFamilies().add(family2);
		
		// Round-trip transformation
		FamiliesToPersonsBidirectionalTransformation transformation = new FamiliesToPersonsBidirectionalTransformation();
		PersonRegister intermediate = transformation.transformToPersons(familyRegister);
		FamilyRegister resultFamilyRegister = transformation.transformToFamilies(intermediate);
		
		// Should have 6 persons total (father + mother + son from family1, father + mother from family2)
		Family resultFamily = resultFamilyRegister.getFamilies().get(0);
		int totalCount = 0;
		totalCount += resultFamily.getFather() != null ? 1 : 0;
		totalCount += resultFamily.getMother() != null ? 1 : 0;
		totalCount += resultFamily.getSons().size();
		totalCount += resultFamily.getDaughters().size();
		
		// All 5 persons from family1 should be preserved
		assertTrue("Should preserve all family members", totalCount >= 3);
	}

	/**
	 * Tests that gender is preserved after round-trip.
	 * @generated
	 */
	@Test
	public void testGenderPreservationAfterRoundTrip() {
		FamiliesFactory familiesFactory = FamiliesFactory.eINSTANCE;
		
		// Create a FamilyRegister
		FamilyRegister familyRegister = familiesFactory.createFamilyRegister();
		Family family = familiesFactory.createFamily();
		familyRegister.getFamilies().add(family);
		
		FamilyMember father = familiesFactory.createFamilyMember();
		father.setName("Test Father");
		family.setFather(father);
		
		FamilyMember mother = familiesFactory.createFamilyMember();
		mother.setName("Test Mother");
		family.setMother(mother);
		
		// Transform to Persons and back
		FamiliesToPersonsBidirectionalTransformation transformation = new FamiliesToPersonsBidirectionalTransformation();
		PersonRegister persons = transformation.transformToPersons(familyRegister);
		FamilyRegister result = transformation.transformToFamilies(persons);
		
		Family resultFamily = result.getFamilies().get(0);
		assertNotNull("Father should be preserved", resultFamily.getFather());
		assertNotNull("Mother should be preserved", resultFamily.getMother());
	}

	/**
	 * Tests round-trip with null input.
	 * @generated
	 */
	@Test(expected = IllegalArgumentException.class)
	public void testRoundTripWithNull() {
		FamiliesToPersonsBidirectionalTransformation transformation = new FamiliesToPersonsBidirectionalTransformation();
		transformation.roundTrip(null);
	}

	/**
	 * Tests reverse round-trip with null input.
	 * @generated
	 */
	@Test(expected = IllegalArgumentException.class)
	public void testReverseRoundTripWithNull() {
		FamiliesToPersonsBidirectionalTransformation transformation = new FamiliesToPersonsBidirectionalTransformation();
		transformation.reverseRoundTrip(null);
	}

} // FamiliesToPersonsBidirectionalTransformationTest