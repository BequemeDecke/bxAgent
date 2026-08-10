package com.example.transformation.families;

import com.example.families.FamilyRegister;
import com.example.persons.PersonRegister;

/**
 * Main transformation service for bidirectional transformation between
 * Families and Persons models.
 * 
 * <p>This service provides a unified interface for both forward and backward
 * transformations, allowing easy switching between directions and configuration
 * of transformation behavior.</p>
 * 
 * <h2>Forward Transformation (Families → Persons)</h2>
 * <p>Transforms a family-based structure into a flat person register.
 * Gender is inferred from FamilyMember role (father/sons → Male, mother/daughters → Female).
 * All persons receive the default birthday "0000-1-1".</p>
 * 
 * <h2>Backward Transformation (Persons → Families)</h2>
 * <p>Transforms a flat person register into a family-based structure.
 * This transformation is lossy because family grouping information is not
 * present in the Persons model. Uses configurable grouping strategy.</p>
 * 
 * @see FamiliesToPersonsTransformation
 * @see PersonsToFamiliesTransformation
 * @see BackwardConfiguration
 */
public class FamiliesPersonsTransformation {
    
    private FamiliesToPersonsTransformation forwardTransformation;
    private PersonsToFamiliesTransformation backwardTransformation;
    
    /**
     * Creates a new transformation service with default configurations.
     */
    public FamiliesPersonsTransformation() {
        this.forwardTransformation = FamiliesToPersonsTransformation.create();
        this.backwardTransformation = PersonsToFamiliesTransformation.create();
    }
    
    /**
     * Transforms a FamilyRegister into a PersonRegister (forward direction).
     * 
     * @param familyRegister The source FamilyRegister to transform
     * @return A new PersonRegister containing all transformed persons
     */
    public PersonRegister toPersons(FamilyRegister familyRegister) {
        return forwardTransformation.transform(familyRegister);
    }
    
    /**
     * Transforms a PersonRegister into a FamilyRegister (backward direction).
     * Uses the configured grouping strategy.
     * 
     * @param personRegister The source PersonRegister to transform
     * @return A new FamilyRegister containing the transformed families
     */
    public FamilyRegister toFamilies(PersonRegister personRegister) {
        return backwardTransformation.transform(personRegister);
    }
    
    /**
     * Gets the forward transformation instance.
     * 
     * @return The FamiliesToPersonsTransformation instance
     */
    public FamiliesToPersonsTransformation getForwardTransformation() {
        return forwardTransformation;
    }
    
    /**
     * Gets the backward transformation instance.
     * 
     * @return The PersonsToFamiliesTransformation instance
     */
    public PersonsToFamiliesTransformation getBackwardTransformation() {
        return backwardTransformation;
    }
    
    /**
     * Sets the default birthday value for forward transformations.
     * 
     * @param birthday The default birthday string
     * @return This transformation service for method chaining
     */
    public FamiliesPersonsTransformation setDefaultBirthday(String birthday) {
        forwardTransformation.setDefaultBirthday(birthday);
        return this;
    }
    
    /**
     * Sets the backward transformation configuration.
     * 
     * @param configuration The BackwardConfiguration instance
     * @return This transformation service for method chaining
     */
    public FamiliesPersonsTransformation setBackwardConfiguration(BackwardConfiguration configuration) {
        backwardTransformation.setConfiguration(configuration);
        return this;
    }
    
    /**
     * Creates a new transformation service with default configurations.
     * 
     * @return A new FamiliesPersonsTransformation instance
     */
    public static FamiliesPersonsTransformation create() {
        return new FamiliesPersonsTransformation();
    }
    
    /**
     * Creates a new transformation service configured for Strategy B
     * (single family for all persons in backward transformation).
     * 
     * @return A new FamiliesPersonsTransformation configured for single family grouping
     */
    public static FamiliesPersonsTransformation createWithSingleFamilyBackward() {
        FamiliesPersonsTransformation transformation = new FamiliesPersonsTransformation();
        transformation.setBackwardConfiguration(BackwardConfiguration.createSingleFamilyConfig());
        return transformation;
    }
}