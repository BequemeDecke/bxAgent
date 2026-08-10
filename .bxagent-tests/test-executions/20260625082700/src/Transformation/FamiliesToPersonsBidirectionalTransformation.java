package Transformation;

import Families.FamilyRegister;
import Persons.PersonRegister;

/**
 * <!-- begin-user-doc -->
 * Provides bidirectional transformation between the Families model and the Persons model.
 * 
 * <p>
 * This service coordinates:
 * <ul>
 *   <li>Forward transformation: {@link FamilyRegister} → {@link PersonRegister}</li>
 *   <li>Backward transformation: {@link PersonRegister} → {@link FamilyRegister}</li>
 * </ul>
 * </p>
 * 
 * <p>
 * <b>Note on Round-Trip Consistency:</b>
 * Due to information loss in the forward transformation (flattening of family structure, 
 * loss of family names, default birthday values), perfect round-trip consistency is not guaranteed.
 * The transformation guarantees:
 * <ul>
 *   <li>All persons are preserved with their names</li>
 *   <li>Gender information is correctly maintained (Male/Female subclasses)</li>
 *   <li>Birthday values are preserved when going backward</li>
 * </ul>
 * Family structure, family names, and specific roles (father/son/mother/daughter) may differ after round-trip.
 * </p>
 * 
 * @see FamiliesToPersonsForwardTransformer
 * @see PersonsToFamiliesBackwardTransformer
 * @generated
 */
public class FamiliesToPersonsBidirectionalTransformation {

	/**
	 * The forward transformer instance.
	 */
	private final FamiliesToPersonsForwardTransformer forwardTransformer;

	/**
	 * The backward transformer instance.
	 */
	private final PersonsToFamiliesBackwardTransformer backwardTransformer;

	/**
	 * Creates a new bidirectional transformation service with default settings.
	 * @generated
	 */
	public FamiliesToPersonsBidirectionalTransformation() {
		this.forwardTransformer = new FamiliesToPersonsForwardTransformer();
		this.backwardTransformer = new PersonsToFamiliesBackwardTransformer();
	}

	/**
	 * Creates a new bidirectional transformation service with specified backward strategy.
	 * 
	 * @param backwardStrategy the strategy to use for backward transformation
	 * @generated
	 */
	public FamiliesToPersonsBidirectionalTransformation(PersonsToFamiliesBackwardTransformer.FamilyReconstructionStrategy backwardStrategy) {
		this.forwardTransformer = new FamiliesToPersonsForwardTransformer();
		this.backwardTransformer = new PersonsToFamiliesBackwardTransformer(backwardStrategy);
	}

	/**
	 * Transforms a {@link FamilyRegister} to a {@link PersonRegister}.
	 * 
	 * @param familyRegister the source FamilyRegister
	 * @return the transformed PersonRegister
	 * @throws IllegalArgumentException if familyRegister is null
	 * @generated
	 */
	public PersonRegister transformToPersons(FamilyRegister familyRegister) {
		return forwardTransformer.transform(familyRegister);
	}

	/**
	 * Transforms a {@link PersonRegister} to a {@link FamilyRegister}.
	 * 
	 * @param personRegister the source PersonRegister
	 * @return the transformed FamilyRegister
	 * @throws IllegalArgumentException if personRegister is null
	 * @generated
	 */
	public FamilyRegister transformToFamilies(PersonRegister personRegister) {
		return backwardTransformer.transform(personRegister);
	}

	/**
	 * Performs a round-trip transformation: Families → Persons → Families.
	 * 
	 * <p>
	 * This method transforms a FamilyRegister to PersonRegister and then back to FamilyRegister.
	 * Due to information loss, the returned FamilyRegister may differ from the original.
	 * </p>
	 * 
	 * @param familyRegister the original FamilyRegister
	 * @return a new FamilyRegister after round-trip transformation
	 * @throws IllegalArgumentException if familyRegister is null
	 * @generated
	 */
	public FamilyRegister roundTrip(FamilyRegister familyRegister) {
		if (familyRegister == null) {
			throw new IllegalArgumentException("FamilyRegister cannot be null");
		}
		
		// Forward transformation
		PersonRegister personRegister = forwardTransformer.transform(familyRegister);
		
		// Backward transformation
		return backwardTransformer.transform(personRegister);
	}

	/**
	 * Performs a reverse round-trip transformation: Persons → Families → Persons.
	 * 
	 * <p>
	 * This method transforms a PersonRegister to FamilyRegister and then back to PersonRegister.
	 * Due to information loss, the returned PersonRegister may differ from the original.
	 * </p>
	 * 
	 * @param personRegister the original PersonRegister
	 * @return a new PersonRegister after round-trip transformation
	 * @throws IllegalArgumentException if personRegister is null
	 * @generated
	 */
	public PersonRegister reverseRoundTrip(PersonRegister personRegister) {
		if (personRegister == null) {
			throw new IllegalArgumentException("PersonRegister cannot be null");
		}
		
		// Backward transformation
		FamilyRegister familyRegister = backwardTransformer.transform(personRegister);
		
		// Forward transformation
		return forwardTransformer.transform(familyRegister);
	}

	/**
	 * Gets the forward transformer.
	 * 
	 * @return the forward transformer instance
	 * @generated
	 */
	public FamiliesToPersonsForwardTransformer getForwardTransformer() {
		return forwardTransformer;
	}

	/**
	 * Gets the backward transformer.
	 * 
	 * @return the backward transformer instance
	 * @generated
	 */
	public PersonsToFamiliesBackwardTransformer getBackwardTransformer() {
		return backwardTransformer;
	}

} // FamiliesToPersonsBidirectionalTransformation