package com.example.transformation.families;

import java.util.IdentityHashMap;
import java.util.Map;

import com.example.families.Family;
import com.example.families.FamilyMember;
import com.example.families.FamilyRegister;
import com.example.persons.Female;
import com.example.persons.Male;
import com.example.persons.Person;
import com.example.persons.PersonRegister;
import com.example.persons.PersonsFactory;

/**
 * Transformation class for converting a family-based structure (Families model)
 * into a flat person register (Persons model).
 * 
 * <p>Forward Transformation: Families → Persons</p>
 * 
 * <p>This transformation handles the following challenges:</p>
 * <ul>
 *   <li>Structure Mismatch: Flattens hierarchical family structure into flat register</li>
 *   <li>Missing Birthday: Uses default value "0000-1-1" for all persons</li>
 *   <li>Gender Inference: Determines gender from FamilyMember role via inverse references</li>
 *   <li>Duplicate Processing: Tracks processed members using object identity</li>
 * </ul>
 * 
 * <p>Gender Inference Rules:</p>
 * <ul>
 *   <li>fatherInverse set → Male</li>
 *   <li>motherInverse set → Female</li>
 *   <li>sonsInverse set → Male</li>
 *   <li>daughtersInverse set → Female</li>
 * </ul>
 */
public class FamiliesToPersonsTransformation {
    
    /**
     * Default birthday value for transformed persons.
     */
    public static final String DEFAULT_BIRTHDAY = "0000-1-1";
    
    private String defaultBirthday = DEFAULT_BIRTHDAY;
    
    /**
     * Transforms a FamilyRegister into a PersonRegister.
     * 
     * <p>All FamilyMember instances from all Family objects are extracted and
     * converted to Person objects (Male or Female based on role) in a single
     * flat PersonRegister.</p>
     * 
     * @param familyRegister The source FamilyRegister to transform
     * @return A new PersonRegister containing all transformed persons
     */
    public PersonRegister transform(FamilyRegister familyRegister) {
        PersonRegister personRegister = PersonsFactory.eINSTANCE.createPersonRegister();
        
        Map<FamilyMember, FamilyMember> processedMembers = new IdentityHashMap<>();
        
        for (Family family : familyRegister.getFamilies()) {
            transformFamily(family, personRegister, processedMembers);
        }
        
        return personRegister;
    }
    
    /**
     * Transforms all FamilyMembers from a single Family into Person objects.
     * 
     * @param family The Family to extract members from
     * @param personRegister The target PersonRegister to populate
     * @param processedMembers Map to track already-processed FamilyMembers
     */
    private void transformFamily(Family family, PersonRegister personRegister, 
                                Map<FamilyMember, FamilyMember> processedMembers) {
        // Process father
        if (family.getFather() != null) {
            processFamilyMember(family.getFather(), personRegister, processedMembers);
        }
        
        // Process mother
        if (family.getMother() != null) {
            processFamilyMember(family.getMother(), personRegister, processedMembers);
        }
        
        // Process sons
        for (FamilyMember son : family.getSons()) {
            processFamilyMember(son, personRegister, processedMembers);
        }
        
        // Process daughters
        for (FamilyMember daughter : family.getDaughters()) {
            processFamilyMember(daughter, personRegister, processedMembers);
        }
    }
    
    /**
     * Processes a single FamilyMember, transforming it to an appropriate Person type.
     * 
     * <p>This method determines gender by checking which inverse reference is set.
     * Uses object identity tracking to avoid processing the same FamilyMember twice.</p>
     * 
     * @param member The FamilyMember to transform
     * @param personRegister The target PersonRegister to populate
     * @param processedMembers Map to track already-processed FamilyMembers
     */
    private void processFamilyMember(FamilyMember member, PersonRegister personRegister,
                                      Map<FamilyMember, FamilyMember> processedMembers) {
        // Skip if already processed
        if (processedMembers.containsKey(member)) {
            return;
        }
        
        processedMembers.put(member, member);
        
        // Determine gender from inverse reference
        Person person = inferGenderAndCreatePerson(member);
        
        // Copy name attribute
        person.setName(member.getName());
        
        // Set default birthday
        person.setBirthday(defaultBirthday);
        
        // Add to person register with proper bidirectional reference
        personRegister.getPersons().add(person);
        person.setPersonsInverse(personRegister);
    }
    
    /**
     * Infers the gender of a FamilyMember based on its inverse reference,
     * and creates the appropriate Person subclass.
     * 
     * <p>Gender inference from inverse reference:</p>
     * <ul>
     *   <li>fatherInverse set → creates Male</li>
     *   <li>motherInverse set → creates Female</li>
     *   <li>sonsInverse set → creates Male</li>
     *   <li>daughtersInverse set → creates Female</li>
     * </ul>
     * 
     * @param member The FamilyMember to infer gender from
     * @return A new Person instance (Male or Female) based on role
     */
    private Person inferGenderAndCreatePerson(FamilyMember member) {
        // Check inverse references to determine role
        if (member.getFatherInverse() != null) {
            return PersonsFactory.eINSTANCE.createMale();
        }
        if (member.getMotherInverse() != null) {
            return PersonsFactory.eINSTANCE.createFemale();
        }
        if (member.getSonsInverse() != null) {
            return PersonsFactory.eINSTANCE.createMale();
        }
        if (member.getDaughtersInverse() != null) {
            return PersonsFactory.eINSTANCE.createFemale();
        }
        
        // Default to Male if no inverse reference is set
        // This should not happen in a well-formed model
        return PersonsFactory.eINSTANCE.createMale();
    }
    
    /**
     * Gets the default birthday value used for transformations.
     * 
     * @return The default birthday string
     */
    public String getDefaultBirthday() {
        return defaultBirthday;
    }
    
    /**
     * Sets the default birthday value to use for transformations.
     * 
     * @param defaultBirthday The default birthday string
     * @return This transformation instance for method chaining
     */
    public FamiliesToPersonsTransformation setDefaultBirthday(String defaultBirthday) {
        this.defaultBirthday = defaultBirthday;
        return this;
    }
    
    /**
     * Creates a new transformation instance with default settings.
     * 
     * @return A new FamiliesToPersonsTransformation instance
     */
    public static FamiliesToPersonsTransformation create() {
        return new FamiliesToPersonsTransformation();
    }
}