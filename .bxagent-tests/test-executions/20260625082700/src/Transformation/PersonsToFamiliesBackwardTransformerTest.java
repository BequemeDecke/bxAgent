package Transformation;

import org.junit.Test;
import static org.junit.Assert.*;

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

import PersonsToFamiliesBackwardTransformer.FamilyReconstructionStrategy;

/**
 * Unit tests for {@link PersonsToFamiliesBackwardTransformer}.
 * 
 * <p>
 * Tests cover:
 * <ul>
 *   <li>Empty PersonRegister transformation</li>
 *   <li>Only males scenario</li>
 *   <li>Only females scenario</li>
 *   <li>Mixed genders scenario</li>
 *   <li>Different reconstruction strategies</li>
 *   <li>Bidirectional link verification</li>
 * </ul>
 * </p>
 * 
 * @see PersonsToFamiliesBackwardTransformer
 * @generated
 */
public class PersonsToFamiliesBackwardTransformerTest {

	/**
	 * Creates a default birthday date.
	 * @return a Date representing the default birthday
	 * @generated
	 */
	private Date createDefaultBirthday() {
		Calendar cal = Calendar.getInstance();
		cal.set(Calendar.YEAR, 0);
		cal.set(Calendar.MONTH, 0);
		cal.set(Calendar.DAY_OF_MONTH, 1);
		return cal.getTime();
	}

	/**
	 * Tests transformation with an empty PersonRegister.
	 * @generated
	 */
	@Test
	public void testTransformEmptyPersonRegister() {
		PersonsFactory personsFactory = PersonsFactory.eINSTANCE;
		PersonRegister personRegister = personsFactory.createPersonRegister();
		
		PersonsToFamiliesBackwardTransformer transformer = new PersonsToFamiliesBackwardTransformer();
		FamilyRegister result = transformer.transform(personRegister);
		
		assertNotNull("FamilyRegister should not be null", result);
		assertEquals("FamilyRegister should have 0 families", 0, result.getFamilies().size());
	}

	/**
	 * Tests transformation with only males (SINGLE_FAMILY strategy).
	 * @generated
	 */
	@Test
	public void testTransformOnlyMales() {
		PersonsFactory personsFactory = PersonsFactory.eINSTANCE;
		
		PersonRegister personRegister = personsFactory.createPersonRegister();
		
		Male male1 = personsFactory.createMale();
		male1.setName("Male One");
		personRegister.getPersons().add(male1);
		
		Male male2 = personsFactory.createMale();
		male2.setName("Male Two");
		personRegister.getPersons().add(male2);
		
		PersonsToFamiliesBackwardTransformer transformer = new PersonsToFamiliesBackwardTransformer(FamilyReconstructionStrategy.SINGLE_FAMILY);
		FamilyRegister result = transformer.transform(personRegister);
		
		assertEquals("Should have 1 family", 1, result.getFamilies().size());
		
		Family family = result.getFamilies().get(0);
		assertNotNull("Father should be set", family.getFather());
		assertEquals("Father should have correct name", "Male One", family.getFather().getName());
		assertEquals("Should have 1 son", 1, family.getSons().size());
		assertEquals("Son should have correct name", "Male Two", family.getSons().get(0).getName());
	}

	/**
	 * Tests transformation with only females (SINGLE_FAMILY strategy).
	 * @generated
	 */
	@Test
	public void testTransformOnlyFemales() {
		PersonsFactory personsFactory = PersonsFactory.eINSTANCE;
		
		PersonRegister personRegister = personsFactory.createPersonRegister();
		
		Female female1 = personsFactory.createFemale();
		female1.setName("Female One");
		personRegister.getPersons().add(female1);
		
		Female female2 = personsFactory.createFemale();
		female2.setName("Female Two");
		personRegister.getPersons().add(female2);
		
		PersonsToFamiliesBackwardTransformer transformer = new PersonsToFamiliesBackwardTransformer(FamilyReconstructionStrategy.SINGLE_FAMILY);
		FamilyRegister result = transformer.transform(personRegister);
		
		assertEquals("Should have 1 family", 1, result.getFamilies().size());
		
		Family family = result.getFamilies().get(0);
		assertNotNull("Mother should be set", family.getMother());
		assertEquals("Mother should have correct name", "Female One", family.getMother().getName());
		assertEquals("Should have 1 daughter", 1, family.getDaughters().size());
		assertEquals("Daughter should have correct name", "Female Two", family.getDaughters().get(0).getName());
	}

	/**
	 * Tests transformation with mixed genders (SINGLE_FAMILY strategy).
	 * @generated
	 */
	@Test
	public void testTransformMixedGenders() {
		PersonsFactory personsFactory = PersonsFactory.eINSTANCE;
		
		PersonRegister personRegister = personsFactory.createPersonRegister();
		
		Male male1 = personsFactory.createMale();
		male1.setName("Father");
		personRegister.getPersons().add(male1);
		
		Male male2 = personsFactory.createMale();
		male2.setName("Son");
		personRegister.getPersons().add(male2);
		
		Female female1 = personsFactory.createFemale();
		female1.setName("Mother");
		personRegister.getPersons().add(female1);
		
		Female female2 = personsFactory.createFemale();
		female2.setName("Daughter");
		personRegister.getPersons().add(female2);
		
		PersonsToFamiliesBackwardTransformer transformer = new PersonsToFamiliesBackwardTransformer(FamilyReconstructionStrategy.SINGLE_FAMILY);
		FamilyRegister result = transformer.transform(personRegister);
		
		assertEquals("Should have 1 family", 1, result.getFamilies().size());
		
		Family family = result.getFamilies().get(0);
		assertEquals("Father name should be correct", "Father", family.getFather().getName());
		assertEquals("Mother name should be correct", "Mother", family.getMother().getName());
		assertEquals("Should have 1 son", 1, family.getSons().size());
		assertEquals("Son name should be correct", "Son", family.getSons().get(0).getName());
		assertEquals("Should have 1 daughter", 1, family.getDaughters().size());
		assertEquals("Daughter name should be correct", "Daughter", family.getDaughters().get(0).getName());
	}

	/**
	 * Tests INDIVIDUAL_FAMILIES strategy.
	 * @generated
	 */
	@Test
	public void testIndividualFamiliesStrategy() {
		PersonsFactory personsFactory = PersonsFactory.eINSTANCE;
		
		PersonRegister personRegister = personsFactory.createPersonRegister();
		
		Male male = personsFactory.createMale();
		male.setName("Single Male");
		personRegister.getPersons().add(male);
		
		Female female = personsFactory.createFemale();
		female.setName("Single Female");
		personRegister.getPersons().add(female);
		
		PersonsToFamiliesBackwardTransformer transformer = new PersonsToFamiliesBackwardTransformer(FamilyReconstructionStrategy.INDIVIDUAL_FAMILIES);
		FamilyRegister result = transformer.transform(personRegister);
		
		assertEquals("Should have 2 families", 2, result.getFamilies().size());
	}

	/**
	 * Tests PAIR_FAMILIES strategy.
	 * @generated
	 */
	@Test
	public void testPairFamiliesStrategy() {
		PersonsFactory personsFactory = PersonsFactory.eINSTANCE;
		
		PersonRegister personRegister = personsFactory.createPersonRegister();
		
		// First pair
		Male male1 = personsFactory.createMale();
		male1.setName("Pair Father");
		personRegister.getPersons().add(male1);
		
		Female female1 = personsFactory.createFemale();
		female1.setName("Pair Mother");
		personRegister.getPersons().add(female1);
		
		// Additional child
		Male male2 = personsFactory.createMale();
		male2.setName("Additional Son");
		personRegister.getPersons().add(male2);
		
		PersonsToFamiliesBackwardTransformer transformer = new PersonsToFamiliesBackwardTransformer(FamilyReconstructionStrategy.PAIR_FAMILIES);
		FamilyRegister result = transformer.transform(personRegister);
		
		assertEquals("Should have 1 family", 1, result.getFamilies().size());
		
		Family family = result.getFamilies().get(0);
		assertEquals("Father name should be correct", "Pair Father", family.getFather().getName());
		assertEquals("Mother name should be correct", "Pair Mother", family.getMother().getName());
		assertEquals("Should have 1 son", 1, family.getSons().size());
		assertEquals("Son name should be correct", "Additional Son", family.getSons().get(0).getName());
	}

	/**
	 * Tests that bidirectional links are established.
	 * @generated
	 */
	@Test
	public void testBidirectionalLinks() {
		PersonsFactory personsFactory = PersonsFactory.eINSTANCE;
		
		PersonRegister personRegister = personsFactory.createPersonRegister();
		
		Male male = personsFactory.createMale();
		male.setName("Test Father");
		personRegister.getPersons().add(male);
		
		PersonsToFamiliesBackwardTransformer transformer = new PersonsToFamiliesBackwardTransformer(FamilyReconstructionStrategy.SINGLE_FAMILY);
		FamilyRegister result = transformer.transform(personRegister);
		
		Family family = result.getFamilies().get(0);
		FamilyMember fatherMember = family.getFather();
		
		// Verify bidirectional link
		assertEquals("Father inverse should link to family", family, fatherMember.getFatherInverse());
		assertEquals("Family should link to father", fatherMember, family.getFather());
	}

	/**
	 * Tests that null PersonRegister throws exception.
	 * @generated
	 */
	@Test(expected = IllegalArgumentException.class)
	public void testNullPersonRegister() {
		PersonsToFamiliesBackwardTransformer transformer = new PersonsToFamiliesBackwardTransformer();
		transformer.transform(null);
	}

} // PersonsToFamiliesBackwardTransformerTest