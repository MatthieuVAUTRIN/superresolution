## The Core Concept
The key insight is that if there was a **reflector at the pixel location:**
- Each transducer would have received a reflection at a specific delay time
- By looking up what was actually measured at those specific delay times
- And adding them together
- We get constructive interference where actual reflectors exist

This is the fundamental principle behind delay-and-sum beamforming.
When we align and sum the signals properly, reflections from real objects add constructively, while noise tends to cancel out.

Example Scenario:

Imagine a reflector at depth 5mm:

- For a centered transducer, the echo might arrive at 6.67 μs (5mm ÷ 1500 m/s × 2)
- For an off-center transducer, it would take slightly longer due to the diagonal path
- When we sum the signals at their correct delay times, the reflector "appears" in our image
- This technique essentially transforms time-domain signals into a spatial image, revealing the locations of reflectors in your imaging field


## Why Objects Appear in the Same Position

When beamforming RF data from different transmission angles (-15° and 0°) and see the same object appearing at the same position in both reconstructed images ->

### Coordinate System and Beamforming Reconstruction
The key insight is that the beamforming algorithm correctly compensates for the different wave propagation paths:

- Each beamformed image represents the same physical space, just interrogated from different angles
- The beamforming algorithm maps each reception back to the physical location where the reflection occurred
- It uses the known transmission angle (-15° or 0°) to calculate the appropriate delays