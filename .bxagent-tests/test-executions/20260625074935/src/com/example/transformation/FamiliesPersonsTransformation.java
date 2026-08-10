package com.example.transformation;

import Families.Family;
import Families.FamilyMember;
import Families.FamilyRegister;
import Families.FamiliesFactory;
import Persons.Female;
import Persons.Male;
import Persons.Person;
import Persons.PersonRegister;
import Persons.PersonsFactory;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Bidirectional transformation between Families and Persons EMF models.
 * 
 * This transformation supports:
 * - Forward transformation (Families → Persons): Transform FamilyRegister with Family 
 *   structures into PersonRegister with flat Person list.
 * - Reverse transformation (Persons → Families): Transform PersonRegister into FamilyRegister
 *   with reconstructed Family structures.
 * 
 * ## Gender Mapping Strategy (Forward)
 * FamilyMember roles are mapped to Person subclasses based on their role in the family:
 * - Father → Male
 * - Mother → Female
 * - Son (in Sons list) → Male
 * - Daughter (in Daughters list) → Female
 * 
 * ## Grouping Strategy (Reverse)
 * For reverse transformation, the grouping strategy determines how flat Person lists 
 * are grouped into Family structures. Supported strategies:
 * - SINGLETON_FAMILIES: Each person becomes their own family
 * - SINGLE_FAMILY: All persons are grouped into one family
 * - PAIR_FAMILIES: Males and females are paired into families (first male with first female, etc.)
 * 
 * @author Transformation Generator
 * @generated
 */
public class FamiliesPersonsTransformation {

	/**
	 * Grouping strategy for reverse transformation (Persons → Families).
	 */
	public enum GroupingStrategy {
		/**
		 * Each person becomes their own family.
		 */
		SINGLETON_FAMILIES,
		
		/**
		 * All persons are grouped into one family.
		 */
		SINGLE_FAMILY,
		
		/**
		 * Males and females are paired into families (first male with first female, etc.).
		 */
		PAIR_FAMILIES
	}

	/**
	 * Configuration for the transformation.
	 */
	public static class TransformationConfig {
		/**
		 * Default birthday value to use when not present in source.
		 * Default: "0000-1-1"
		 */
		private Date defaultBirthday;
		
		/**
		 * Grouping strategy for reverse transformation.
		 * Default: SINGLETON_FAMILIES
		 */
		private GroupingStrategy groupingStrategy;
		
		public TransformationConfig() {
			Calendar cal = Calendar.getInstance();
			cal.set(1, 0, 1, 0, 0, 0);
			this.defaultBirthday = cal.getTime();
			this.groupingStrategy = GroupingStrategy.SINGLETON_FAMILIES;
		}
		
		public Date getDefaultBirthday() {
			return defaultBirthday;
		}
		
		public void setDefaultBirthday(Date defaultBirthday) {
			this.defaultBirthday = defaultBirthday;
		}
		
		public GroupingStrategy getGroupingStrategy() {
			return groupingStrategy;
		}
		
		public void setGroupingStrategy(GroupingStrategy groupingStrategy) {
			this.groupingStrategy = groupingStrategy;
		}
	}

	/**
	 * Transformation result containing both the transformed object and mappings.
	 */
	public static class TransformationResult<T> {
		private final T target;
		private final Map<Object, Object> sourceToTargetMapping;
		private final Map<Object, Object> targetToSourceMapping;
		
		public TransformationResult(T target, Map<Object, Object> sourceToTargetMapping, 
								  Map<Object, Object> targetToSourceMapping) {
			this.target = target;
			this.sourceToTargetMapping = sourceToTargetMapping;
			this.targetToSourceMapping = targetToSourceMapping;
		}
		
		public T getTarget() {
			return target;
		}
		
		public Map<Object, Object> getSourceToTargetMapping() {
			return sourceToTargetMapping;
		}
		
		public Map<Object, Object> getTargetToSourceMapping() {
			return targetToSourceMapping;
		}
	}

	private final TransformationConfig config;

	/**
	 * Creates a new transformation with default configuration.
	 */
	public FamiliesPersonsTransformation() {
		this.config = new TransformationConfig();
	}

	/**
	 * Creates a new transformation with specified configuration.
	 * 
	 * @param config the transformation configuration
	 */
	public FamiliesPersonsTransformation(TransformationConfig config) {
		this.config = config;
	}

	/**
	 * Transforms a FamilyRegister into a PersonRegister.
	 * 
	 * **Step 1.1**: Create PersonRegister from FamilyRegister
	 * **Step 1.2**: Iterate through all Families in the source FamilyRegister
	 * **Step 1.3**: Transform each FamilyMember to a Person (Male/Female based on role)
	 * **Step 1.4**: Handle duplicate FamilyMembers
	 * 
	 * @param source the source FamilyRegister
	 * @return the transformation result containing the PersonRegister and mappings
	 */
	public TransformationResult<PersonRegister> transformToTarget(FamilyRegister source) {
		if (source == null) {
			throw new IllegalArgumentException("Source FamilyRegister cannot be null");
		}
		
		Map<Object, Object> sourceToTargetMapping = new IdentityHashMap<>();
		Map<Object, Object> targetToSourceMapping = new IdentityHashMap<>();
		
		// Step 1.1: Create PersonRegister from FamilyRegister
		PersonRegister target = PersonsFactory.eINSTANCE.createPersonRegister();
		
		// Step 1.2: Iterate through all Families in the source FamilyRegister
		for (Family family : source.getFamilies()) {
			transformFamily(family, target, sourceToTargetMapping, targetToSourceMapping);
		}
		
		return new TransformationResult<>(target, sourceToTargetMapping, targetToSourceMapping);
	}

	/**
	 * Transforms a single Family into Person objects.
	 * 
	 * @param family the source Family
	 * @param personRegister the target PersonRegister
	 * @param sourceToTargetMapping mapping from source to target objects
	 * @param targetToSourceMapping mapping from target to source objects
	 */
	private void transformFamily(Family family, PersonRegister personRegister,
								 Map<Object, Object> sourceToTargetMapping,
								 Map<Object, Object> targetToSourceMapping) {
		// Transform father
		FamilyMember father = family.getFather();
		if (father != null) {
			transformFamilyMember(father, true, personRegister, sourceToTargetMapping, targetToSourceMapping);
		}
		
		// Transform mother
		FamilyMember mother = family.getMother();
		if (mother != null) {
			transformFamilyMember(mother, false, personRegister, sourceToTargetMapping, targetToSourceMapping);
		}
		
		// Transform sons
		for (FamilyMember son : family.getSons()) {
			transformFamilyMember(son, true, personRegister, sourceToTargetMapping, targetToSourceMapping);
		}
		
		// Transform daughters
		for (FamilyMember daughter : family.getDaughters()) {
			transformFamilyMember(daughter, false, personRegister, sourceToTargetMapping, targetToSourceMapping);
		}
	}

	/**
	 * Transforms a single FamilyMember into a Person (Male or Female).
	 * 
	 * @param member the source FamilyMember
	 * @param isMale true if the member should be a Male, false for Female
	 * @param personRegister the target PersonRegister
	 * @param sourceToTargetMapping mapping from source to target objects
	 * @param targetToSourceMapping mapping from target to source objects
	 */
	private void transformFamilyMember(FamilyMember member, boolean isMale, PersonRegister personRegister,
								   Map<Object, Object> sourceToTargetMapping,
								   Map<Object, Object> targetToSourceMapping) {
		// Step 1.4: Handle duplicate FamilyMembers
		if (sourceToTargetMapping.containsKey(member)) {
			// Already transformed, skip
			return;
		}
		
		// Step 1.3: Transform each FamilyMember to a Person
		Person person;
		if (isMale) {
			person = PersonsFactory.eINSTANCE.createMale();
		} else {
			person = PersonsFactory.eINSTANCE.createFemale();
		}
		
		// Copy the Name attribute
		person.setName(member.getName());
		
		// Set Birthday to default value
		person.setBirthday(config.getDefaultBirthday());
		
		// Add to PersonRegister (establishes bidirectional link)
		personRegister.getPersons().add(person);
		
		// Update mappings
		sourceToTargetMapping.put(member, person);
		targetToSourceMapping.put(person, member);
	}

	/**
	 * Transforms a PersonRegister into a FamilyRegister.
	 * 
	 * **Step 2.1**: Create FamilyRegister from PersonRegister
	 * **Step 2.2**: Group Persons into Families based on strategy
	 * **Step 2.3**: Reconstruct Family structure for each group
	 * **Step 2.4**: Handle role assignment ambiguity
	 * 
	 * @param source the source PersonRegister
	 * @return the transformation result containing the FamilyRegister and mappings
	 */
	public TransformationResult<FamilyRegister> transformToSource(PersonRegister source) {
		if (source == null) {
			throw new IllegalArgumentException("Source PersonRegister cannot be null");
		}
		
		Map<Object, Object> sourceToTargetMapping = new IdentityHashMap<>();
		Map<Object, Object> targetToSourceMapping = new IdentityHashMap<>();
		
		// Step 2.1: Create FamilyRegister from PersonRegister
		FamilyRegister target = FamiliesFactory.eINSTANCE.createFamilyRegister();
		
		// Step 2.2: Group Persons into Families based on strategy
		List<List<Person>> groups = groupPersons(source.getPersons(), config.getGroupingStrategy());
		
		// Step 2.3: Reconstruct Family structure for each group
		for (List<Person> group : groups) {
			transformGroupToFamily(group, target, sourceToTargetMapping, targetToSourceMapping);
		}
		
		return new TransformationResult<>(target, sourceToTargetMapping, targetToSourceMapping);
	}

	/**
	 * Groups persons based on the specified grouping strategy.
	 * 
	 * @param persons the list of persons to group
	 * @param strategy the grouping strategy
	 * @return list of groups, where each group is a list of persons
	 */
	private List<List<Person>> groupPersons(List<Person> persons, GroupingStrategy strategy) {
		List<List<Person>> groups = new ArrayList<>();
		
		switch (strategy) {
			case SINGLETON_FAMILIES:
				// Each person becomes their own family
				for (Person person : persons) {
					List<Person> singleton = new ArrayList<>();
					singleton.add(person);
					groups.add(singleton);
				}
				break;
				
			case SINGLE_FAMILY:
				// All persons are grouped into one family
				groups.add(new ArrayList<>(persons));
				break;
				
			case PAIR_FAMILIES:
				// Males and females are paired into families
				List<Person> males = new ArrayList<>();
				List<Person> females = new ArrayList<>();
				
				for (Person person : persons) {
					if (person instanceof Male) {
						males.add(person);
					} else if (person instanceof Female) {
						females.add(person);
					}
				}
				
				int maxPairs = Math.max(males.size(), females.size());
				for (int i = 0; i < maxPairs; i++) {
					List<Person> group = new ArrayList<>();
					if (i < males.size()) {
						group.add(males.get(i));
					}
					if (i < females.size()) {
						group.add(females.get(i));
					}
					if (!group.isEmpty()) {
						groups.add(group);
					}
				}
				break;
		}
		
		return groups;
	}

	/**
	 * Transforms a group of persons into a Family.
	 * 
	 * @param group the group of persons
	 * @param familyRegister the target FamilyRegister
	 * @param sourceToTargetMapping mapping from source to target objects
	 * @param targetToSourceMapping mapping from target to source objects
	 */
	private void transformGroupToFamily(List<Person> group, FamilyRegister familyRegister,
									 Map<Object, Object> sourceToTargetMapping,
									 Map<Object, Object> targetToSourceMapping) {
		// Create Family
		Family family = FamiliesFactory.eINSTANCE.createFamily();
		
		// Step 2.4: Handle role assignment ambiguity
		// First male → Father, remaining males → Sons
		// First female → Mother, remaining females → Daughters
		FamilyMember fatherMember = null;
		FamilyMember motherMember = null;
		List<FamilyMember> sons = new ArrayList<>();
		List<FamilyMember> daughters = new ArrayList<>();
		
		List<Male> males = new ArrayList<>();
		List<Female> females = new ArrayList<>();
		
		for (Person person : group) {
			if (person instanceof Male) {
				males.add((Male) person);
			} else if (person instanceof Female) {
				females.add((Female) person);
			}
		}
		
		// Assign roles
		boolean firstMale = true;
		boolean firstFemale = true;
		
		for (Male male : males) {
			FamilyMember member = transformPersonToFamilyMember(male, sourceToTargetMapping, targetToSourceMapping);
			if (firstMale) {
				fatherMember = member;
				firstMale = false;
			} else {
				sons.add(member);
			}
		}
		
		for (Female female : females) {
			FamilyMember member = transformPersonToFamilyMember(female, sourceToTargetMapping, targetToSourceMapping);
			if (firstFemale) {
				motherMember = member;
				firstFemale = false;
			} else {
				daughters.add(member);
			}
		}
		
		// Set Family name (derived from members)
		String familyName = deriveFamilyName(fatherMember, motherMember, sons, daughters);
		family.setName(familyName);
		
		// Set containment references
		if (fatherMember != null) {
			family.setFather(fatherMember);
		}
		if (motherMember != null) {
			family.setMother(motherMember);
		}
		for (FamilyMember son : sons) {
			family.getSons().add(son);
		}
		for (FamilyMember daughter : daughters) {
			family.getDaughters().add(daughter);
		}
		
		// Add Family to FamilyRegister
		familyRegister.getFamilies().add(family);
	}

	/**
	 * Derives a family name from family members.
	 * 
	 * @param father the father member (or null)
	 * @param mother the mother member (or null)
	 * @param sons list of sons
	 * @param daughters list of daughters
	 * @return the derived family name
	 */
	private String deriveFamilyName(FamilyMember father, FamilyMember mother, 
								 List<FamilyMember> sons, List<FamilyMember> daughters) {
		StringBuilder sb = new StringBuilder();
		
		if (father != null) {
			sb.append(father.getName());
		}
		if (mother != null) {
			if (sb.length() > 0) {
				sb.append(" & ");
			}
			sb.append(mother.getName());
		}
		
		if (sb.length() == 0) {
			// Fallback: use children's names
			if (!sons.isEmpty()) {
				sb.append(sons.get(0).getName());
			} else if (!daughters.isEmpty()) {
				sb.append(daughters.get(0).getName());
			}
		}
		
		return sb.length() > 0 ? sb.toString() : "Unknown Family";
	}

	/**
	 * Transforms a Person into a FamilyMember.
	 * 
	 * @param person the source Person
	 * @param sourceToTargetMapping mapping from source to target objects
	 * @param targetToSourceMapping mapping from target to source objects
	 * @return the created FamilyMember
	 */
	private FamilyMember transformPersonToFamilyMember(Person person,
													 Map<Object, Object> sourceToTargetMapping,
													 Map<Object, Object> targetToSourceMapping) {
		FamilyMember member = FamiliesFactory.eINSTANCE.createFamilyMember();
		member.setName(person.getName());
		
		sourceToTargetMapping.put(person, member);
		targetToSourceMapping.put(member, person);
		
		return member;
	}

	/**
	 * Gets the transformation configuration.
	 * 
	 * @return the configuration
	 */
	public TransformationConfig getConfig() {
		return config;
	}
} // FamiliesPersonsTransformation