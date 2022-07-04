Summary
===============================

Summary of progress in May 2022

- **Ideas for setting up training procedure**
    1. Introduce a **third label**. The responses for the two classes (appr/ retr) oscillate. *We can define a segment as time points between the intersections of the response curves.* This is because at intersection influence of both the stimuli is similar, and at other time points one of them dominates. Thus such a segment would have least *contamination* from response for the other stimulus. Since it may be difficult to determine which class the fMRI responses at the intersection time points belong to, we can introduce a third target label: **Don't care, or indistinguishable, or unidentifiable**. So the time points between intersections can belong to either of the two classes and the points around intersections can belong to the new third class. 

- **Progress**
    1. While generating results for the first idea I learned the hard way that it is cumbersome working with custom (loss/training) functions in tensorflow. I shifted the entire codebase to PyTorch. 
    PyTorch has an extension called `captum`, latin for comprehension, that has all the standard model interpretation methods available in one place.

    2. The emoprox2 dataset has stimulus oscillating similar to simusoids making it unintuitive to decide the approach and retreat time segments. So we shifted to another paradigm, MAX. MAX has a simple block design wherein disjoint blocks of threat and safe conditions are presented to the partiicpant in a pseudorandom fashion. Each block is of the same duration, 16.25 seconds. As this data is cleaner to work with, we will experiment here first and then probably continue investigating emoprox2 again.
