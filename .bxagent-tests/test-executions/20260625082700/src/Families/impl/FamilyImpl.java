package Families.impl;

import java.util.Date;
import java.util.ArrayList;
import java.util.List;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.BasicEObjectImpl;
import org.eclipse.emf.ecore.impl.EObjectImpl;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;
import org.eclipse.emf.ecore.util.EObjectWithInverseResolvingEList;
import org.eclipse.emf.ecore.util.EcoreUtil;
import org.eclipse.emf.ecore.util.InternalEList;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Family</b></em>'.
 * <!-- end-user-doc -->
 *
 * @generated
 */
public class FamilyImpl extends MinimalEObjectImpl implements Family {
	/**
	 * @generated
	 */
	protected FamilyImpl() {
		super();
	}

	/**
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return FamiliesPackageImpl.Literals.FAMILY;
	}

	/**
	 * @generated
	 */
	protected FamilyMember father;

	/**
	 * @generated
	 */
	@Override
	public FamilyMember getFather() {
		if (father != null && father.eIsProxy()) {
			InternalEObject o = (InternalEObject) father;
			father = (FamilyMember) eResolveProxy(o);
		}
		return father;
	}

	/**
	 * @generated
	 */
	@Override
	public FamilyMember basicGetFather() {
		return father;
	}

	/**
	 * @generated
	 */
	public NotificationChain basicSetFather(FamilyMember newFather, Notifications msgs) {
		FamilyMember oldFather = father;
		father = newFather;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notifications.SET, FamiliesPackageImpl.FAMILY__FATHER, oldFather, newFather);
			if (msgs == null) {
				deliverChangeNotification(notification);
			} else {
				msgs.add(notification);
			}
		}
		return msgs;
	}

	/**
	 * @generated
	 */
	@Override
	public void setFather(FamilyMember newFather) {
		if (newFather != null) {
			if (((InternalEObject) newFather).eIsProxy()) {
				setFather((FamilyMember) eResolveProxy((InternalEObject) newFather));
				return;
			}
		}
		if (father != newFather) {
			Notifications msgs = null;
			if (father != null) {
				msgs = ((InternalEObject) father).eInverseRemove(this, FamiliesPackageImpl.FAMILY_MEMBER__FATHER_INVERSE, null, msgs);
			}
			if (newFather != null) {
				msgs = ((InternalEObject) newFather).eInverseAdd(this, FamiliesPackageImpl.FAMILY_MEMBER__FATHER_INVERSE, null, msgs);
			}
			msgs = basicSetFather(newFather, msgs);
		} else if (eNotificationRequired()) {
			eNotify(new ENotificationImpl(this, Notifications.SET, FamiliesPackageImpl.FAMILY__FATHER, newFather, newFather));
		}
	}

	/**
	 * @generated
	 */
	protected FamilyMember mother;

	/**
	 * @generated
	 */
	@Override
	public FamilyMember getMother() {
		if (mother != null && mother.eIsProxy()) {
			InternalEObject o = (InternalEObject) mother;
			mother = (FamilyMember) eResolveProxy(o);
		}
		return mother;
	}

	/**
	 * @generated
	 */
	@Override
	public FamilyMember basicGetMother() {
		return mother;
	}

	/**
	 * @generated
	 */
	public NotificationChain basicSetMother(FamilyMember newMother, Notifications msgs) {
		FamilyMember oldMother = mother;
		mother = newMother;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notifications.SET, FamiliesPackageImpl.FAMILY__MOTHER, oldMother, newMother);
			if (msgs == null) {
				deliverChangeNotification(notification);
			} else {
				msgs.add(notification);
			}
		}
		return msgs;
	}

	/**
	 * @generated
	 */
	@Override
	public void setMother(FamilyMember newMother) {
		if (newMother != null) {
			if (((InternalEObject) newMother).eIsProxy()) {
				setMother((FamilyMember) eResolveProxy((InternalEObject) newMother));
				return;
			}
		}
		if (mother != newMother) {
			Notifications msgs = null;
			if (mother != null) {
				msgs = ((InternalEObject) mother).eInverseRemove(this, FamiliesPackageImpl.FAMILY_MEMBER__MOTHER_INVERSE, null, msgs);
			}
			if (newMother != null) {
				msgs = ((InternalEObject) newMother).eInverseAdd(this, FamiliesPackageImpl.FAMILY_MEMBER__MOTHER_INVERSE, null, msgs);
			}
			msgs = basicSetMother(newMother, msgs);
		} else if (eNotificationRequired()) {
			eNotify(new ENotificationImpl(this, Notifications.SET, FamiliesPackageImpl.FAMILY__MOTHER, newMother, newMother));
		}
	}

	/**
	 * @generated
	 */
	protected EList<FamilyMember> sons;

	/**
	 * @generated
	 */
	@Override
	public EList<FamilyMember> getSons() {
		if (sons == null) {
			sons = new EObjectWithInverseResolvingEList<FamilyMember>(FamilyMember.class, this, FamiliesPackageImpl.FAMILY__SONS, FamiliesPackageImpl.FAMILY_MEMBER__SONS_INVERSE);
		}
		return sons;
	}

	/**
	 * @generated
	 */
	protected EList<FamilyMember> daughters;

	/**
	 * @generated
	 */
	@Override
	public EList<FamilyMember> getDaughters() {
		if (daughters == null) {
			daughters = new EObjectWithInverseResolvingEList<FamilyMember>(FamilyMember.class, this, FamiliesPackageImpl.FAMILY__DAUGHTERS, FamiliesPackageImpl.FAMILY_MEMBER__DAUGHTERS_INVERSE);
		}
		return daughters;
	}

	/**
	 * @generated
	 */
	protected String name;

	/**
	 * @generated
	 */
	@Override
	public String getName() {
		return name;
	}

	/**
	 * @generated
	 */
	@Override
	public void setName(String newName) {
		name = newName;
	}

	/**
	 * @generated
	 */
	protected FamilyRegister familiesInverse;

	/**
	 * @generated
	 */
	@Override
	public FamilyRegister getFamiliesInverse() {
		if (familiesInverse != null && familiesInverse.eIsProxy()) {
			InternalEObject o = (InternalEObject) familiesInverse;
			familiesInverse = (FamilyRegister) eResolveProxy(o);
		}
		return familiesInverse;
	}

	/**
	 * @generated
	 */
	@Override
	public FamilyRegister basicGetFamiliesInverse() {
		return familiesInverse;
	}

	/**
	 * @generated
	 */
	public NotificationChain basicSetFamiliesInverse(FamilyRegister newFamiliesInverse, Notifications msgs) {
		FamilyRegister oldFamiliesInverse = familiesInverse;
		familiesInverse = newFamiliesInverse;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notifications.SET, FamiliesPackageImpl.FAMILY__FAMILIES_INVERSE, oldFamiliesInverse, newFamiliesInverse);
			if (msgs == null) {
				deliverChangeNotification(notification);
			} else {
				msgs.add(notification);
			}
		}
		return msgs;
	}

	/**
	 * @generated
	 */
	@Override
	public void setFamiliesInverse(FamilyRegister newFamiliesInverse) {
		if (newFamiliesInverse != null) {
			if (((InternalEObject) newFamiliesInverse).eIsProxy()) {
				setFamiliesInverse((FamilyRegister) eResolveProxy((InternalEObject) newFamiliesInverse));
				return;
			}
		}
		if (familiesInverse != newFamiliesInverse) {
			Notifications msgs = null;
			if (familiesInverse != null) {
				msgs = ((InternalEObject) familiesInverse).eInverseRemove(this, FamiliesPackageImpl.FAMILY_REGISTER__FAMILIES, null, msgs);
			}
			if (newFamiliesInverse != null) {
				msgs = ((InternalEObject) newFamiliesInverse).eInverseAdd(this, FamiliesPackageImpl.FAMILY_REGISTER__FAMILIES, null, msgs);
			}
			msgs = basicSetFamiliesInverse(newFamiliesInverse, msgs);
		} else if (eNotificationRequired()) {
			eNotify(new ENotificationImpl(this, Notifications.SET, FamiliesPackageImpl.FAMILY__FAMILIES_INVERSE, newFamiliesInverse, newFamiliesInverse));
		}
	}

	/**
	 * @generated
	 */
	@Override
	public NotificationChain eInverseAdd(InternalEObject otherEnd, int featureID, Notifications msgs) {
		switch (featureID) {
			case FamiliesPackageImpl.FAMILY__FATHER:
				if (father != null) {
					msgs = ((InternalEObject) father).eInverseRemove(this, FamiliesPackageImpl.FAMILY_MEMBER__FATHER_INVERSE, null, msgs);
				}
				return basicSetFather((FamilyMember) otherEnd, msgs);
			case FamiliesPackageImpl.FAMILY__MOTHER:
				if (mother != null) {
					msgs = ((InternalEObject) mother).eInverseRemove(this, FamiliesPackageImpl.FAMILY_MEMBER__MOTHER_INVERSE, null, msgs);
				}
				return basicSetMother((FamilyMember) otherEnd, msgs);
			case FamiliesPackageImpl.FAMILY__SONS:
				return ((InternalEObject) otherEnd).eInverseAdd(this, FamiliesPackageImpl.FAMILY_MEMBER__SONS_INVERSE, null, msgs);
			case FamiliesPackageImpl.FAMILY__DAUGHTERS:
				return ((InternalEObject) otherEnd).eInverseAdd(this, FamiliesPackageImpl.FAMILY_MEMBER__DAUGHTERS_INVERSE, null, msgs);
			case FamiliesPackageImpl.FAMILY__FAMILIES_INVERSE:
				if (familiesInverse != null) {
					msgs = ((InternalEObject) familiesInverse).eInverseRemove(this, FamiliesPackageImpl.FAMILY_REGISTER__FAMILIES, null, msgs);
				}
				return basicSetFamiliesInverse((FamilyRegister) otherEnd, msgs);
		}
		return super.eInverseAdd(otherEnd, featureID, msgs);
	}

	/**
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, Notifications msgs) {
		switch (featureID) {
			case FamiliesPackageImpl.FAMILY__FATHER:
				msgs = ((InternalEObject) father).eInverseRemove(this, FamiliesPackageImpl.FAMILY_MEMBER__FATHER_INVERSE, null, msgs);
				return basicSetFather(null, msgs);
			case FamiliesPackageImpl.FAMILY__MOTHER:
				msgs = ((InternalEObject) mother).eInverseRemove(this, FamiliesPackageImpl.FAMILY_MEMBER__MOTHER_INVERSE, null, msgs);
				return basicSetMother(null, msgs);
			case FamiliesPackageImpl.FAMILY__SONS:
				return ((InternalEObject) otherEnd).eInverseRemove(this, FamiliesPackageImpl.FAMILY_MEMBER__SONS_INVERSE, null, msgs);
			case FamiliesPackageImpl.FAMILY__DAUGHTERS:
				return ((InternalEObject) otherEnd).eInverseRemove(this, FamiliesPackageImpl.FAMILY_MEMBER__DAUGHTERS_INVERSE, null, msgs);
			case FamiliesPackageImpl.FAMILY__FAMILIES_INVERSE:
				msgs = ((InternalEObject) familiesInverse).eInverseRemove(this, FamiliesPackageImpl.FAMILY_REGISTER__FAMILIES, null, msgs);
				return basicSetFamiliesInverse(null, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	/**
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case FamiliesPackageImpl.FAMILY__FATHER:
				if (resolve) {
					return getFather();
				}
				return basicGetFather();
			case FamiliesPackageImpl.FAMILY__MOTHER:
				if (resolve) {
					return getMother();
				}
				return basicGetMother();
			case FamiliesPackageImpl.FAMILY__SONS:
				if (resolve) {
					return getSons();
				}
				return getSons().list();
			case FamiliesPackageImpl.FAMILY__DAUGHTERS:
				if (resolve) {
					return getDaughters();
				}
				return getDaughters().list();
			case FamiliesPackageImpl.FAMILY__NAME:
				return getName();
			case FamiliesPackageImpl.FAMILY__FAMILIES_INVERSE:
				if (resolve) {
					return getFamiliesInverse();
				}
				return basicGetFamiliesInverse();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * @generated
	 */
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case FamiliesPackageImpl.FAMILY__FATHER:
				setFather((FamilyMember) newValue);
				return;
			case FamiliesPackageImpl.FAMILY__MOTHER:
				setMother((FamilyMember) newValue);
				return;
			case FamiliesPackageImpl.FAMILY__SONS:
				getSons().clear();
				getSons().addAll((EList<FamilyMember>) newValue);
				return;
			case FamiliesPackageImpl.FAMILY__DAUGHTERS:
				getDaughters().clear();
				getDaughters().addAll((EList<FamilyMember>) newValue);
				return;
			case FamiliesPackageImpl.FAMILY__NAME:
				setName((String) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	/**
	 * @generated
	 */
	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case FamiliesPackageImpl.FAMILY__FATHER:
				setFather((FamilyMember) NOT_SET);
				return;
			case FamiliesPackageImpl.FAMILY__MOTHER:
				setMother((FamilyMember) NOT_SET);
				return;
			case FamiliesPackageImpl.FAMILY__SONS:
				getSons().clear();
				return;
			case FamiliesPackageImpl.FAMILY__DAUGHTERS:
				getDaughters().clear();
				return;
			case FamiliesPackageImpl.FAMILY__NAME:
				setName(NOT_SET);
				return;
		}
		super.eUnset(featureID);
	}

	/**
	 * @generated
	 */
	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case FamiliesPackageImpl.FAMILY__FATHER:
				return father != null;
			case FamiliesPackageImpl.FAMILY__MOTHER:
				return mother != null;
			case FamiliesPackageImpl.FAMILY__SONS:
				return sons != null && !sons.isEmpty();
			case FamiliesPackageImpl.FAMILY__DAUGHTERS:
				return daughters != null && !daughters.isEmpty();
			case FamiliesPackageImpl.FAMILY__NAME:
				return name != NOT_SET;
			case FamiliesPackageImpl.FAMILY__FAMILIES_INVERSE:
				return familiesInverse != null;
		}
		return super.eIsSet(featureID);
	}
} // FamilyImpl