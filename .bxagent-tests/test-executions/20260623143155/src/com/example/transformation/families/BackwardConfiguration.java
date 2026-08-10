package com.example.transformation.families;

import com.example.families.FamiliesFactory;
import com.example.families.Family;
import com.example.families.FamilyMember;
import com.example.families.FamilyRegister;
import com.example.persons.Person;
import com.example.persons.PersonRegister;

/**
 * Configuration class for the backward transformation (Persons → Families).
 * 
 * This class provides configurable options for how to handle the family grouping
 * when transforming a flat PersonRegister back into a hierarchical Families structure.
 * 
 * Strategy A (default): One Family Per Person
 * - Each person becomes their own family unit with appropriate role
 * - Males are assigned as fathers
 * - Females are assigned as mothers
 * 
 * Strategy B: Single Family for All Persons
 * - Creates one Family containing all persons
 * - Requires logic to balance father/mother/sons/daughters assignments
 */
public class BackwardConfiguration {
    
    /**
     * Enum representing the available family grouping strategies.
     */
    public enum GroupingStrategy {
        /**
         * Strategy A: One Family Per Person.
         * Each person becomes their own family unit.
         */
        ONE_FAMILY_PER_PERSON,
        
        /**
         * Strategy B: Single Family for All Persons.
         * All persons are grouped into one family.
         */
        SINGLE_FAMILY_FOR_ALL
    }
    
    private GroupingStrategy strategy = GroupingStrategy.ONE_FAMILY_PER_PERSON;
    private String defaultBirthday = "0000-1-1";
    
    /**
     * Gets the currently configured grouping strategy.
     * 
     * @return The grouping strategy to use
     */
    public GroupingStrategy getStrategy() {
        return strategy;
    }
    
    /**
     * Sets the grouping strategy to use.
     * 
     * @param strategy The grouping strategy
     * @return This configuration for method chaining
     */
    public BackwardConfiguration setStrategy(GroupingStrategy strategy) {
        this.strategy = strategy;
        return this;
    }
    
    /**
     * Gets the default birthday value to use for FamilyMembers.
     * 
     * @return The default birthday string
     */
    public String getDefaultBirthday() {
        return defaultBirthday;
    }
    
    /**
     * Sets the default birthday value to use for FamilyMembers.
     * Note: FamilyMember may not have a birthday attribute, but this
     * is provided for consistency if needed.
     * 
     * @param defaultBirthday The default birthday string
     * @return This configuration for method chaining
     */
    public BackwardConfiguration setDefaultBirthday(String defaultBirthday) {
        this.defaultBirthday = defaultBirthday;
        return this;
    }
    
    /**
     * Creates a FamilyRegister with the configured grouping strategy.
     * 
     * @param personRegister The source PersonRegister to transform
     * @return A new FamilyRegister containing the transformed families
     */
    public FamilyRegister createFamilyRegister(PersonRegister personRegister) {
        FamilyRegister familyRegister = FamiliesFactory.eINSTANCE.createFamilyRegister();
        
        switch (strategy) {
            case ONE_FAMILY_PER_PERSON:
                createOneFamilyPerPerson(personRegister, familyRegister);
                break;
            case SINGLE_FAMILY_FOR_ALL:
                createSingleFamilyForAll(personRegister, familyRegister);
                break;
        }
        
        return familyRegister;
    }
    
    /**
     * Strategy A: Create one Family per Person.
     * Each person becomes their own family unit with appropriate role:
     * - Males are assigned as father
     * - Females are assigned as mother
     * 
     * @param personRegister The source PersonRegister
     * @param familyRegister The target FamilyRegister to populate
     */
    private void createOneFamilyPerPerson(PersonRegister personRegister, FamilyRegister familyRegister) {
        for (Person person : personRegister.getPersons()) {
            Family family = FamiliesFactory.eINSTANCE.createFamily();
            family.setName(person.getName() + "'s Family");
            
            FamilyMember member = FamiliesFactory.eINSTANCE.createFamilyMember();
            member.setName(person.getName());
            
            if (person instanceof com.example.persons.Male) {
                family.setFather(member);
                member.setFatherInverse(family);
            } else if (person instanceof com.example.persons.Female) {
                family.setMother(member);
                member.setMotherInverse(family);
            }
            
            familyRegister.getFamilies().add(member);
            familyRegister.getFamilies().add(family);
        }
    }
    
    /**
     * Strategy B: Create a single Family containing all persons.
     * Distributes persons into roles based on gender:
     * - First male becomes father
     * - First female becomes mother
     * - Remaining males become sons
     * - Remaining females become daughters
     * 
     * @param personRegister The source PersonRegister
     * @param familyRegister The target FamilyRegister to populate
     */
    private void createSingleFamilyForAll(PersonRegister personRegister, FamilyRegister familyRegister) {
        Family family = FamiliesFactory.eINSTANCE.createFamily();
        family.setName("Combined Family");
        
        FamilyMember fatherMember = null;
        FamilyMember motherMember = null;
        
        for (Person person : personRegister.getPersons()) {
            FamilyMember member = FamiliesFactory.eINSTANCE.createFamilyMember();
            member.setName(person.getName());
            
            if (person instanceof com.example.persons.Male) {
                if (fatherMember == null) {
                    fatherMember = member;
                    family.setFather(member);
                    member.setFatherInverse(family);
                } else {
                    family.getSons().add(member);
                    member.setSonsInverse(family);
                }
            } else if (person instanceof com.example.persons.Female) {
                if (motherMember == null) {
                    motherMember = member;
                    family.setMother(member);
                    member.setMotherInverse(family);
                } else {
                    family.getDaughters().add(member);
                    member.setDaughtersInverse(family);
                }
            }
        }
        
        familyRegister.getFamilies().add(family);
    }
    
    /**
     * Creates a default configuration with Strategy A.
     * 
     * @return A new BackwardConfiguration with default settings
     */
    public static BackwardConfiguration createDefault() {
        return new BackwardConfiguration();
    }
    
    /**
     * Creates a configuration with Strategy B (single family for all).
     * 
     * @return A new BackwardConfiguration configured for single family grouping
     */
    public static BackwardConfiguration createSingleFamilyConfig() {
        return new BackwardConfiguration()
                .setStrategy(GroupingStrategy.SINGLE_FAMILY_FOR_ALL);
    }
}