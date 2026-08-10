package com.example.transformation.families;

import java.util.IdentityHashMap;
import java.util.Map;

import com.example.families.Family;
import com.example.families.FamilyMember;
import com.example.families.FamilyRegister;
import com.example.families.FamiliesFactory;
import com.example.persons.Female;
import com.example.persons.Male;
import com.example.persons.Person;
import com.example.persons.PersonRegister;

/**
 * Transformation class for converting a flat person register (Persons model)
 * into a family-based structure (Families model).
 * 
 * <p>Backward Transformation: Persons → Families</p>
 * 
 * <p>This transformation is lossy because the flat PersonRegister structure
 * does not contain information about family groupings. The transformation
 * uses a configurable strategy to create family structures.</p>
 * 
 * <p>Available Strategies:</p>
 * <ul>
 *   <li>ONE_FAMILY_PER_PERSON (default): Each person becomes their own family unit</li>
 *   <li>SINGLE_FAMILY_FOR_ALL: All persons are grouped into one family</li>
 * </ul>
 * 
 * <p>Note: This transformation cannot restore the original family structure
 * due to information loss during the forward transformation.</p>
 */
public class PersonsToFamiliesTransformation {
    
    private BackwardConfiguration configuration;
    
    /**
     * Creates a new transformation with the default configuration.
     * Uses Strategy A (ONE_FAMILY_PER_PERSON) by default.
     */
    public PersonsToFamiliesTransformation() {
        this.configuration = BackwardConfiguration.createDefault();
    }
    
    /**
     * Creates a new transformation with a custom configuration.
     * 
     * @param configuration The configuration specifying the grouping strategy
     */
    public PersonsToFamiliesTransformation(BackwardConfiguration configuration) {
        this.configuration = configuration;
    }
    
    /**
     * Transforms a PersonRegister into a FamilyRegister using the configured strategy.
     * 
     * @param personRegister The source PersonRegister to transform
     * @return A new FamilyRegister containing the transformed families
     */
    public FamilyRegister transform(PersonRegister personRegister) {
        return configuration.createFamilyRegister(personRegister);
    }
    
    /**
     * Transforms a PersonRegister into a FamilyRegister using Strategy A
     * (one family per person).
     * 
     * <p>Each person becomes their own family unit:</p>
     * <ul>
     *   <li>Males are assigned as father</li>
     *   <li>Females are assigned as mother</li>
     * </ul>
     * 
     * @param personRegister The source PersonRegister to transform
     * @return A new FamilyRegister with one family per person
     */
    public FamilyRegister transformAsOneFamilyPerPerson(PersonRegister personRegister) {
        return BackwardConfiguration.createDefault().createFamilyRegister(personRegister);
    }
    
    /**
     * Transforms a PersonRegister into a FamilyRegister using Strategy B
     * (single family for all persons).
     * 
     * <p>All persons are grouped into one family:</p>
     * <ul>
     *   <li>First male becomes father</li>
     *   <li>First female becomes mother</li>
     *   <li>Remaining males become sons</li>
     *   <li>Remaining females become daughters</li>
     * </ul>
     * 
     * @param personRegister The source PersonRegister to transform
     * @return A new FamilyRegister with a single combined family
     */
    public FamilyRegister transformAsSingleFamily(PersonRegister personRegister) {
        return BackwardConfiguration.createSingleFamilyConfig().createFamilyRegister(personRegister);
    }
    
    /**
     * Gets the configuration used by this transformation.
     * 
     * @return The BackwardConfiguration instance
     */
    public BackwardConfiguration getConfiguration() {
        return configuration;
    }
    
    /**
     * Sets the configuration to use for this transformation.
     * 
     * @param configuration The BackwardConfiguration instance
     * @return This transformation instance for method chaining
     */
    public PersonsToFamiliesTransformation setConfiguration(BackwardConfiguration configuration) {
        this.configuration = configuration;
        return this;
    }
    
    /**
     * Creates a new transformation instance with default settings.
     * 
     * @return A new PersonsToFamiliesTransformation instance
     */
    public static PersonsToFamiliesTransformation create() {
        return new PersonsToFamiliesTransformation();
    }
    
    /**
     * Creates a new transformation instance with Strategy B configuration.
     * 
     * @return A new PersonsToFamiliesTransformation configured for single family
     */
    public static PersonsToFamiliesTransformation createSingleFamily() {
        return new PersonsToFamiliesTransformation(BackwardConfiguration.createSingleFamilyConfig());
    }
}