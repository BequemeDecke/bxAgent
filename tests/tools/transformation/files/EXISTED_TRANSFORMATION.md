---
source_model_package: de.hof-university.models.family
target_model_package: de.hof-university.models.person
iteration: 0
---

# Transformation Plan

## 1. Model Implementations

The implementation of the source model looks as followed:
--- BEGIN SOURCE MODEL ---
```java
class Family {
    private Member father;
    private Member mother;
}
```
--- END SOURCE MODEL ---

The implementation of the target model looks as followed:
--- BEGIN TARGET MODEL ---
```java
class Person {
    private String birthday;
}
```
--- END TARGET MODEL ---

## 2. Transformation Direction

Bidirectional transformation is required, meaning source to target and target to source:
--- BEGIN TRANSFORMATION DIRECTION ---
Just a simple batch transformation in the beginning!
--- END TRANSFORMATION DIRECTION ---

---

## 3. Identified Difficulties

Several difficulties with the transformation itself have been identified:
--- BEGIN DIFFICULTIES ---
1. **Naming of a FamilyMember**: The exact name is lost. 
--- END DIFFICULTIES ---

Please note that you have to provide Configuration parameters if there are multiple strategies to resolve a difficulty.

---

## 4. Implementation Steps

--- BEGIN IMPLEMENTATION STEPS ---
1. Create the Configuration
2. Implement the transformation logic
--- END IMPLEMENTATION STEPS ---
