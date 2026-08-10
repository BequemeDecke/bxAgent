---
source_model_package: Families
target_model_package: Persons
iteration: 1
---

# Transformation Plan

## 1. Model Implementations

The implementation of the source model looks as followed:
--- BEGIN SOURCE MODEL ---
The source model (Families) consists of three main classes:
- FamilyRegister: Root container holding a list of Family objects via 'families' containment reference
- Family: Represents a family unit with name attribute, and containment references for father, mother, sons (list), and daughters (list), all of type FamilyMember
- FamilyMember: Represents an individual with name attribute and bidirectional inverse references (fatherInverse, motherInverse, sonsInverse, daughtersInverse) back to Family

Key characteristics:
- Hierarchical structure with Family as the organizing unit
- Gender is implicit through role (father=male, mother=female, sons=male, daughters=female)
- Bidirectional references maintain consistency between Family and FamilyMember
- A FamilyMember can only belong to one family (container relationship)
--- END SOURCE MODEL ---

The implementation of the target model looks as followed:
--- BEGIN TARGET MODEL ---
The target model (Persons) consists of three main classes:
- PersonRegister: Root container holding a list of Person objects via 'persons' containment reference
- Person: Abstract base class with name attribute, birthday attribute (default "0000-1-1"), and bidirectional inverse reference to PersonRegister
- Male and Female: Concrete subclasses of Person (indicated by factory methods createMale() and createFemale())

Key characteristics:
- Flat structure with no family grouping
- Explicit gender through class hierarchy (Male/Female)
- Birthday attribute required but not present in source model
- All persons are peers in the register without relational structure
--- END TARGET MODEL ---

## 2. Transformation Direction

Bidirectional transformation is required, meaning source to target and target to source:
--- BEGIN TRANSFORMATION DIRECTION ---
Bidirectional transformation required:
- Forward (Families → Persons): Transform family-based structure into flat person register, extracting gender from role and generating default birthdays
- Backward (Persons → Families): Transform flat person register into family-based structure, but this requires grouping persons into families which involves inference or configuration since no family relationship exists in target model
--- END TRANSFORMATION DIRECTION ---

---

## 3. Identified Difficulties

Several difficulties with the transformation itself have been identified:
--- BEGIN DIFFICULTIES ---
## 1. Structure Mismatch - Hierarchical to Flat (Forward)
**Challenge**: The source model organizes persons within Family units, while the target model is a flat register of persons without any grouping.
**Impact**: Forward transformation loses family relationship information. All FamilyMembers from all families will be flattened into a single PersonRegister.
**Resolution Strategy**: Extract all FamilyMember instances from all Family objects and create corresponding Person objects (Male/Female based on role) in the PersonRegister. Family context is lost.

## 2. Missing Birthday Attribute (Forward)
**Challenge**: The target Person class requires a birthday attribute (type Date, default "0000-1-1"), but the source FamilyMember has no equivalent field.
**Impact**: Cannot preserve birthday information in forward direction.
**Resolution Strategy**: Use the default value "0000-1-1" for all generated Person objects, or allow configuration to specify a default date.

## 3. Gender Inference from Role (Forward)
**Challenge**: Source model encodes gender implicitly through role (father→male, mother→female, sons→male, daughters→female), while target uses explicit class hierarchy (Male/Female subclasses).
**Impact**: Must correctly infer gender based on which containment reference the FamilyMember appears in within its parent Family.
**Resolution Strategy**: Check the inverse references (fatherInverse, motherInverse, sonsInverse, daughtersInverse) to determine role and thus gender. A FamilyMember with fatherInverse set becomes Male; motherInverse becomes Female; sonsInverse becomes Male; daughtersInverse becomes Female.

## 4. Irreversible Structure Loss (Backward)
**Challenge**: The most significant difficulty. The target model has no information about family groupings. When transforming Persons → Families, there is no way to reconstruct the original family structure.
**Impact**: Backward transformation cannot restore original Family organization. This makes true bidirectional round-trip consistency impossible without additional metadata.
**Resolution Strategy Options** (requires configuration):
   - **Strategy A**: Create one Family per Person (each person becomes their own family unit with appropriate role)
   - **Strategy B**: Create a single Family containing all persons (assign roles arbitrarily or based on gender distribution)
   - **Strategy C**: Require external configuration file specifying desired family groupings
   - **Strategy D**: Mark backward transformation as lossy/unavailable
   
**Recommended Default**: Strategy A (one family per person) as it preserves individual identity, though semantics differ from original.

## 5. Name Collisions in Flattened Structure
**Challenge**: Multiple FamilyMembers with the same name can exist across different families in the source. When flattened, the target register may contain duplicate names.
**Impact**: Potential ambiguity when transforming backward if relying on name for matching.
**Resolution Strategy**: Do not rely solely on name for identity matching. Consider using object identity or generate unique identifiers if needed for round-trip consistency.

## 6. Container Reference Management
**Challenge**: Both models use EMF bidirectional container references that must be kept consistent.
**Impact**: Incorrect reference management can lead to dangling references or inconsistent state.
**Resolution Strategy**: Ensure proper setting of inverse references (personsInverse in target, various *Inverse references in source) during creation and maintain consistency when modifying relationships.
--- END DIFFICULTIES ---

Please note that you have to provide Configuration parameters if there are multiple strategies to resolve a difficulty.

---

## 4. Implementation Steps

--- BEGIN IMPLEMENTATION STEPS ---
# Forward Transformation (Families → Persons)

## Step 1.1: Create Target PersonRegister
- Instantiate a new PersonRegister object using PersonsFactory
- This will serve as the root container for all transformed persons

## Step 1.2: Iterate Through Source Families
- Access the FamilyRegister from the source model
- Iterate through each Family in the families list
- For each Family, process all contained FamilyMembers

## Step 1.3: Extract and Transform Each FamilyMember
For each FamilyMember found:
- Check which inverse reference is set to determine role:
  - If `fatherInverse` is set → gender is Male
  - If `motherInverse` is set → gender is Female  
  - If `sonsInverse` is set → gender is Male
  - If `daughtersInverse` is set → gender is Female
- Use PersonsFactory to create appropriate subclass:
  - Create Male instance for male roles
  - Create Female instance for female roles
- Copy the name attribute from FamilyMember to Person
- Set birthday to default value "0000-1-1" (or configured default)
- Add the Person to the PersonRegister's persons list
- Set the personsInverse reference to maintain bidirectional consistency

## Step 1.4: Handle Duplicate Processing
- Ensure each FamilyMember is only processed once (a member could theoretically be accessed through multiple paths)
- Track processed members using object identity to avoid duplicates in the target register

---

# Backward Transformation (Persons → Families)

**Configuration Required**: Select family grouping strategy (default: one family per person)

## Step 2.1: Create Source FamilyRegister
- Instantiate a new FamilyRegister using FamiliesFactory
- This will serve as the root container for all reconstructed families

## Step 2.2: Apply Configured Grouping Strategy

### Strategy A (Default): One Family Per Person
- For each Person in the PersonRegister:
  - Create a new Family object
  - Assign a generated or default family name (e.g., "[PersonName]'s Family")
  - Based on Person type (Male/Female), assign an appropriate role:
    - Male → assign as father OR son (requires additional logic/configuration)
    - Female → assign as mother OR daughter (requires additional logic/configuration)
  - Create corresponding FamilyMember with same name
  - Establish proper containment and inverse references
  - Add Family to FamilyRegister

### Strategy B: Single Family for All Persons
- Create one Family object
- Distribute persons into roles based on gender counts
- Requires logic to balance father/mother/sons/daughters assignments

### Strategy C: External Configuration
- Read family grouping specification from configuration file
- Map persons to families according to specification

## Step 2.3: Create FamilyMembers with Proper Roles
For each person-to-role assignment:
- Use FamiliesFactory to create FamilyMember
- Copy name from Person to FamilyMember
- Set appropriate containment reference in Family (father, mother, sons, or daughters)
- Set corresponding inverse reference in FamilyMember
- For list-type references (sons, daughters), add to the appropriate EList

## Step 2.4: Maintain Bidirectional Consistency
- Ensure all inverse references are properly set:
  - fatherInverse ↔ father
  - motherInverse ↔ mother
  - sonsInverse ↔ sons
  - daughtersInverse ↔ daughters
  - familiesInverse ↔ families
- Verify EMF containment relationships are correctly established
--- END IMPLEMENTATION STEPS ---