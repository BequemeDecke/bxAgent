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

/**
 * Unit tests for {@link FamiliesToPersonsForwardTransformer}.
 * 
 * <p>
 * Tests cover:
 * <ul>
 *   <li>Empty FamilyRegister transformation</li>
 *   <li>Single family with all member types</li>
 *   <li>Multiple families</li>
 *   <li>Families with missing members</li>
 *   <li>Name preservation verification</li>
 *   <li>Birthday default value verification</li>
 * </ul>
 * </p>
 * 
 * @see FamiliesToPersonsForwardTransformer
 * @generated
 */
public class FamiliesToPersonsForwardTransformerTest {

	/**
	 * Tests transformation with an empty FamilyRegister.
	 * @generated
	 */
	@Test
	public void testTransformEmptyFamilyRegister() {
		FamiliesFactory familiesFactory = FamiliesFactory.eINSTANCE;
		FamilyRegister familyRegister = familiesFactory.createFamilyRegister();
		
		FamiliesToPersonsForwardTransformer transformer = new FamiliesToPersonsForwardTransformer();
		PersonRegister result = transformer.transform(familyRegister);
		
		assertNotNull("PersonRegister should not be null", result);
		assertEquals("PersonRegister should be empty", 0, result.getPersons().size());
	}

	/**
	 * Tests transformation with a single family containing all member types.
	 * @generated
	 */
	@Test
	public void testTransformSingleFamilyWithAllMembers() {
		FamiliesFactory familiesFactory = FamiliesFactory.eINSTANCE;
		PersonsFactory personsFactory = PersonsFactory.eINSTANCE;
		
		// Create a FamilyRegister with one family
		FamilyRegister familyRegister = familiesFactory.createFamilyRegister();
		Family family = familiesFactory.createFamily();
		familyRegister.getFamilies().add(family);
		
		// Create father
		FamilyMember father = familiesFactory.createFamilyMember();
		father.setName("John Smith");
		family.setFather(father);
		
		// Create mother
		FamilyMember mother = familiesFactory.createFamilyMember();
		mother.setName("Jane Smith");
		family.setMother(mother);
		
		// Create sons
		FamilyMember son1 = familiesFactory.createFamilyMember();
		son1.setName("Bob Smith");
		family.getSons().add(son1);
		
		FamilyMember son2 = familiesFactory.createFamilyMember();
		son2.setName("Mike Smith");
		family.getSons().add(son2);
		
		// Create daughters
		FamilyMember daughter1 = familiesFactory.createFamilyMember();
		daughter1.setName("Mary Smith");
		family.getDaughters().add(daughter1);
		
		FamilyMember daughter2 = familiesFactory.createFamilyMember();
		daughter2.setName("Anna Smith");
		family.getDaughters().add(daughter2);
		
		// Transform
		FamiliesToPersonsForwardTransformer transformer = new FamiliesToPersonsForwardTransformer();
		PersonRegister result = transformer.transform(familyRegister);
		
		// Verify total count
		assertEquals("Should have 6 persons", 6, result.getPersons().size());
		
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
		
		assertEquals("Should have 3 males (father + 2 sons)", 3, maleCount);
		assertEquals("Should have 3 females (mother + 2 daughters)", 3, femaleCount);
	}

	/**
	 * Tests transformation with multiple families.
	 * @generated
	 */
	@Test
	public void testTransformMultipleFamilies() {
		FamiliesFactory familiesFactory = FamiliesFactory.eINSTANCE;
		
		// Create FamilyRegister with two families
		FamilyRegister familyRegister = familiesFactory.createFamilyRegister();
		
		// Family 1
		Family family1 = familiesFactory.createFamily();
		family1.setName("Family 1");
		FamilyMember father1 = familiesFactory.createFamilyMember();
		father1.setName("Father One");
		family1.setFather(father1);
		FamilyMember mother1 = familiesFactory.createFamilyMember();
		mother1.setName("Mother One");
		family1.setMother(mother1);
		familyRegister.getFamilies().add(family1);
		
		// Family 2
		Family family2 = familiesFactory.createFamily();
		family2.setName("Family 2");
		FamilyMember father2 = familiesFactory.createFamilyMember();
		father2.setName("Father Two");
		family2.setFather(father2);
		FamilyMember mother2 = familiesFactory.createFamilyMember();
		mother2.setName("Mother Two");
		family2.setMother(mother2);
		familyRegister.getFamilies().add(family2);
		
		// Transform
		FamiliesToPersonsForwardTransformer transformer = new FamiliesToPersonsForwardTransformer();
		PersonRegister result = transformer.transform(familyRegister);
		
		// Should have 4 persons total
		assertEquals("Should have 4 persons", 4, result.getPersons().size());
	}

	/**
	 * Tests transformation with missing father.
	 * @generated
	 */
	@Test
	public void testTransformMissingFather() {
		FamiliesFactory familiesFactory = FamiliesFactory.eINSTANCE;
		
		FamilyRegister familyRegister = familiesFactory.createFamilyRegister();
		Family family = familiesFactory.createFamily();
		familyRegister.getFamilies().add(family);
		
		// Only mother and son
		FamilyMember mother = familiesFactory.createFamilyMember();
		mother.setName("Single Mother");
		family.setMother(mother);
		
		FamilyMember son = familiesFactory.createFamilyMember();
		son.setName("Only Son");
		family.getSons().add(son);
		
		// Transform
		FamiliesToPersonsForwardTransformer transformer = new FamiliesToPersonsForwardTransformer();
		PersonRegister result = transformer.transform(familyRegister);
		
		assertEquals("Should have 2 persons", 2, result.getPersons().size());
	}

	/**
	 * Tests transformation with no children.
	 * @generated
	 */
	@Test
	public void testTransformNoChildren() {
		FamiliesFactory familiesFactory = FamiliesFactory.eINSTANCE;
		
		FamilyRegister familyRegister = familiesFactory.createFamilyRegister();
		Family family = familiesFactory.createFamily();
		familyRegister.getFamilies().add(family);
		
		// Only parents, no children
		FamilyMember father = familiesFactory.createFamilyMember();
		father.setName("Father Only");
		family.setFather(father);
		
		FamilyMember mother = familiesFactory.createFamilyMember();
		mother.setName("Mother Only");
		family.setMother(mother);
		
		// Transform
		FamiliesToPersonsForwardTransformer transformer = new FamiliesToPersonsForwardTransformer();
		PersonRegister result = transformer.transform(familyRegister);
		
		assertEquals("Should have 2 persons", 2, result.getPersons().size());
	}

	/**
	 * Tests that names are preserved correctly.
	 * @generated
	 */
	@Test
	public void testNamePreservation() {
		FamiliesFactory familiesFactory = FamiliesFactory.eINSTANCE;
		
		FamilyRegister familyRegister = familiesFactory.createFamilyRegister();
		Family family = familiesFactory.createFamily();
		familyRegister.getFamilies().add(family);
		
		FamilyMember father = familiesFactory.createFamilyMember();
		father.setName("Unique Father Name");
		family.setFather(father);
		
		FamilyMember mother = familiesFactory.createFamilyMember();
		mother.setName("Unique Mother Name");
		family.setMother(mother);
		
		// Transform
		FamiliesToPersonsForwardTransformer transformer = new FamiliesToPersonsForwardTransformer();
		PersonRegister result = transformer.transform(familyRegister);
		
		// Verify names
		boolean foundFather = false;
		boolean foundMother = false;
		for (Person person : result.getPersons()) {
			if ("Unique Father Name".equals(person.getName())) {
				foundFather = true;
				assertTrue("Father should be Male", person instanceof Male);
			} else if ("Unique Mother Name".equals(person.getName())) {
				foundMother = true;
				assertTrue("Mother should be Female", person instanceof Female);
			}
		}
		
		assertTrue("Father name should be preserved", foundFather);
		assertTrue("Mother name should be preserved", foundMother);
	}

	/**
	 * Tests that birthday defaults are set correctly.
	 * @generated
	 */
	@Test
	public void testBirthdayDefaults() {
		FamiliesFactory familiesFactory = FamiliesFactory.eINSTANCE;
		
		FamilyRegister familyRegister = familiesFactory.createFamilyRegister();
		Family family = familiesFactory.createFamily();
		familyRegister.getFamilies().add(family);
		
		FamilyMember father = familiesFactory.createFamilyMember();
		father.setName("Test Father");
		family.setFather(father);
		
		// Transform
		FamiliesToPersonsForwardTransformer transformer = new FamiliesToPersonsForwardTransformer();
		PersonRegister result = transformer.transform(familyRegister);
		
		assertEquals("Should have 1 person", 1, result.getPersons().size());
		assertEquals("Birthday should be default", FamiliesToPersonsForwardTransformer.DEFAULT_BIRTHDAY, result.getPersons().get(0).getBirthday());
	}

	/**
	 * Tests that null FamilyRegister throws exception.
	 * @generated
	 */
	@Test(expected = IllegalArgumentException.class)
	public void testNullFamilyRegister() {
		FamiliesToPersonsForwardTransformer transformer = new FamiliesToPersonsForwardTransformer();
		transformer.transform(null);
	}

} // FamiliesToPersonsForwardTransformerTest