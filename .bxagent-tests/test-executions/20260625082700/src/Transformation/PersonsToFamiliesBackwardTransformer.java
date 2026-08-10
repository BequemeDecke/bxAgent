package Transformation;

import java.util.ArrayList;
import java.util.List;

import Families.FamilyRegister;
import Families.Family;
import Families.FamilyMember;
import Families.FamiliesFactory;

import Persons.PersonRegister;
import Persons.Person;
import Persons.Male;
import Persons.Female;

/**
 * <!-- begin-user-doc -->
 * Transforms a {@link PersonRegister} containing {@link Person} objects (Male or Female)
 * into a {@link FamilyRegister} containing {@link Family} objects with {@link FamilyMember}s.
 * 
 * This is the backward transformation from the Persons model to the Families model.
 * 
 * <p>
 * The transformation strategy (Configuration Option A - Single Family):
 * <ul>
 *   <li>Creates a single Family containing all transformed persons</li>
 *   <li>Assigns the first male as father, remaining males as sons</li>
 *   <li>Assigns the first female as mother, remaining females as daughters</li>
 *   <li>Establishes bidirectional links between FamilyMember and Family</li>
 *   <li>Sets family name to a default value (family structure cannot be perfectly reconstructed)</li>
 * </ul>
 * </p>
 * 
 * <p>
 * <b>Note on Information Loss:</b>
 * Due to the flattening nature of the forward transformation, perfect round-trip
 * consistency is impossible. This backward transformation uses heuristics to reconstruct
 * family structure, which may differ from the original.
 * </p>
 * 
 * @see FamiliesToPersonsForwardTransformer
 * @see FamiliesToPersonsBidirectionalTransformation
 * @generated
 */
public class PersonsToFamiliesBackwardTransformer {

	/**
	 * Strategy for family reconstruction in backward transformation.
	 * @generated
	 */
	public enum FamilyReconstructionStrategy {
		/**
		 * Create a single family containing all persons.
		 * First male becomes father, remaining males become sons.
		 * First female becomes mother, remaining females become daughters.
		 */
		SINGLE_FAMILY,
		
		/**
		 * Create individual families for each person.
		 * No family relationships are created.
		 */
		INDIVIDUAL_FAMILIES,
		
		/**
		 * Create pairs: each male and female form a family unit.
		 * Remaining males/females become children in the first family.
		 */
		PAIR_FAMILIES
	}

	/**
	 * The default strategy used when no strategy is specified.
	 * @generated
	 */
	public static final FamilyReconstructionStrategy DEFAULT_STRATEGY = FamilyReconstructionStrategy.SINGLE_FAMILY;

	/**
	 * Default family name used when creating family units.
	 * @generated
	 */
	public static final String DEFAULT_FAMILY_NAME = "Reconstructed Family";

	/**
	 * The strategy to use for family reconstruction.
	 */
	private FamilyReconstructionStrategy strategy;

	/**
	 * Creates a new transformer with the default strategy.
	 * @generated
	 */
	public PersonsToFamiliesBackwardTransformer() {
		this(DEFAULT_STRATEGY);
	}

	/**
	 * Creates a new transformer with the specified strategy.
	 * 
	 * @param strategy the strategy to use for family reconstruction
	 * @generated
	 */
	public PersonsToFamiliesBackwardTransformer(FamilyReconstructionStrategy strategy) {
		this.strategy = strategy;
	}

	/**
	 * Transforms the given {@link PersonRegister} into a {@link FamilyRegister}.
	 * 
	 * @param personRegister the source PersonRegister to transform
	 * @return a new FamilyRegister containing all transformed families
	 * @throws IllegalArgumentException if personRegister is null
	 * @generated
	 */
	public FamilyRegister transform(PersonRegister personRegister) {
		if (personRegister == null) {
			throw new IllegalArgumentException("PersonRegister cannot be null");
		}
		
		// Step 2.1: Create FamilyRegister from PersonRegister
		FamiliesFactory familiesFactory = FamiliesFactory.eINSTANCE;
		FamilyRegister familyRegister = familiesFactory.createFamilyRegister();
		
		// Step 2.3: Separate Persons by Gender
		List<Male> males = new ArrayList<Male>();
		List<Female> females = new ArrayList<Female>();
		
		for (Person person : personRegister.getPersons()) {
			if (person instanceof Male) {
				males.add((Male) person);
			} else if (person instanceof Female) {
				females.add((Female) person);
			}
		}
		
		// Step 2.4: Reconstruct Family Structure based on strategy
		switch (strategy) {
			case SINGLE_FAMILY:
				reconstructSingleFamily(familyRegister, males, females);
				break;
			case INDIVIDUAL_FAMILIES:
				reconstructIndividualFamilies(familyRegister, males, females);
				break;
			case PAIR_FAMILIES:
				reconstructPairFamilies(familyRegister, males, females);
				break;
		}
		
		// Step 2.6: Complete Backward Transformation
		return familyRegister;
	}

	/**
	 * Reconstructs using the SINGLE_FAMILY strategy.
	 * Creates a single family containing all persons.
	 * 
	 * @param familyRegister the target FamilyRegister
	 * @param males list of males to assign
	 * @param females list of females to assign
	 * @generated
	 */
	private void reconstructSingleFamily(FamilyRegister familyRegister, List<Male> males, List<Female> females) {
		FamiliesFactory familiesFactory = FamiliesFactory.eINSTANCE;
		
		// Create a single Family
		Family family = familiesFactory.createFamily();
		family.setName(DEFAULT_FAMILY_NAME);
		
		// Assign father (first male if available)
		FamilyMember fatherMember = null;
		if (!males.isEmpty()) {
			fatherMember = createFamilyMember(family, males.get(0));
			family.setFather(fatherMember);
		}
		
		// Assign sons (remaining males)
		for (int i = 1; i < males.size(); i++) {
			FamilyMember sonMember = createFamilyMember(family, males.get(i));
			family.getSons().add(sonMember);
		}
		
		// Assign mother (first female if available)
		FamilyMember motherMember = null;
		if (!females.isEmpty()) {
			motherMember = createFamilyMember(family, females.get(0));
			family.setMother(motherMember);
		}
		
		// Assign daughters (remaining females)
		for (int i = 1; i < females.size(); i++) {
			FamilyMember daughterMember = createFamilyMember(family, females.get(i));
			family.getDaughters().add(daughterMember);
		}
		
		// Step 2.5: Establish Bidirectional Links
		if (fatherMember != null) {
			fatherMember.setFatherInverse(family);
		}
		if (motherMember != null) {
			motherMember.setMotherInverse(family);
		}
		for (FamilyMember son : family.getSons()) {
			son.setSonsInverse(family);
		}
		for (FamilyMember daughter : family.getDaughters()) {
			daughter.setDaughtersInverse(family);
		}
		
		// Add family to FamilyRegister
		familyRegister.getFamilies().add(family);
		family.setFamiliesInverse(familyRegister);
	}

	/**
	 * Reconstructs using the INDIVIDUAL_FAMILIES strategy.
	 * Creates individual family units for each person.
	 * 
	 * @param familyRegister the target FamilyRegister
	 * @param males list of males to assign
	 * @param females list of females to assign
	 * @generated
	 */
	private void reconstructIndividualFamilies(FamilyRegister familyRegister, List<Male> males, List<Female> females) {
		FamiliesFactory familiesFactory = FamiliesFactory.eINSTANCE;
		
		// Create family units for each male
		for (Male male : males) {
			Family family = familiesFactory.createFamily();
			family.setName("Family for " + male.getName());
			
			FamilyMember fatherMember = createFamilyMember(family, male);
			family.setFather(fatherMember);
			fatherMember.setFatherInverse(family);
			
			familyRegister.getFamilies().add(family);
			family.setFamiliesInverse(familyRegister);
		}
		
		// Create family units for each female
		for (Female female : females) {
			Family family = familiesFactory.createFamily();
			family.setName("Family for " + female.getName());
			
			FamilyMember motherMember = createFamilyMember(family, female);
			family.setMother(motherMember);
			motherMember.setMotherInverse(family);
			
			familyRegister.getFamilies().add(family);
			family.setFamiliesInverse(familyRegister);
		}
	}

	/**
	 * Reconstructs using the PAIR_FAMILIES strategy.
	 * Pairs males and females as parents, remaining become children.
	 * 
	 * @param familyRegister the target FamilyRegister
	 * @param males list of males to assign
	 * @param females list of females to assign
	 * @generated
	 */
	private void reconstructPairFamilies(FamilyRegister familyRegister, List<Male> males, List<Female> females) {
		FamiliesFactory familiesFactory = FamiliesFactory.eINSTANCE;
		
		Family family = familiesFactory.createFamily();
		family.setName(DEFAULT_FAMILY_NAME);
		
		// First male as father
		FamilyMember fatherMember = null;
		if (!males.isEmpty()) {
			fatherMember = createFamilyMember(family, males.get(0));
			family.setFather(fatherMember);
			fatherMember.setFatherInverse(family);
		}
		
		// First female as mother
		FamilyMember motherMember = null;
		if (!females.isEmpty()) {
			motherMember = createFamilyMember(family, females.get(0));
			family.setMother(motherMember);
			motherMember.setMotherInverse(family);
		}
		
		// Remaining males as sons
		for (int i = 1; i < males.size(); i++) {
			FamilyMember sonMember = createFamilyMember(family, males.get(i));
			family.getSons().add(sonMember);
			sonMember.setSonsInverse(family);
		}
		
		// Remaining females as daughters
		for (int i = 1; i < females.size(); i++) {
			FamilyMember daughterMember = createFamilyMember(family, females.get(i));
			family.getDaughters().add(daughterMember);
			daughterMember.setDaughtersInverse(family);
		}
		
		familyRegister.getFamilies().add(family);
		family.setFamiliesInverse(familyRegister);
	}

	/**
	 * Creates a FamilyMember from a Person.
	 * 
	 * @param family the target family
	 * @param person the source person
	 * @return a new FamilyMember with the name from the person
	 * @generated
	 */
	private FamilyMember createFamilyMember(Family family, Person person) {
		FamiliesFactory familiesFactory = FamiliesFactory.eINSTANCE;
		FamilyMember familyMember = familiesFactory.createFamilyMember();
		familyMember.setName(person.getName());
		return familyMember;
	}

	/**
	 * Gets the current strategy.
	 * 
	 * @return the strategy being used
	 * @generated
	 */
	public FamilyReconstructionStrategy getStrategy() {
		return strategy;
	}

	/**
	 * Sets the strategy to use for family reconstruction.
	 * 
	 * @param strategy the new strategy
	 * @generated
	 */
	public void setStrategy(FamilyReconstructionStrategy strategy) {
		this.strategy = strategy;
	}

} // PersonsToFamiliesBackwardTransformer