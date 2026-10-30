# Examples

Synthetic test material for H3 Create. Every person, animal and place here is AI-generated: the pictures with Qwen-Image 2.1, the
videos with MiniMax H3 (three to six shots per clip, the same character in every shot). No real person's likeness was used.
Offered under CC BY 4.0 for testing and showing the app, subject to the licenses of the models that made them (see ../THIRD_PARTY.md).

`library/` is the default Motion Library (H3C_LIBRARY). Load a clip from the Motion Library, add the pictures, paste the prompt:

| Clip | Pictures | Prompt | What it tests |
|---|---|---|---|
| `p04_two_friends_snow.mp4` | `p04_new_man.jpg`, `p04_new_woman.jpg`, `p04_desert_village.jpg` | `Replace the man with @image1 and the woman with @image2, and move them to the desert village in @image3.` | two people + a new place, a people-free drone shot |
| `p01_street_to_stage.mp4` | `p01_new_woman.jpg` | `Replace the woman with @image1.` | a costume change between shots (street clothes → flamenco dress) |
| `p02_beach_long_hair.mp4` | `p02_buzzcut_man.jpg` | `Replace the woman with the man from @image1.` | long hair flying around → a buzz cut (the check retakes the part if the hair leaks in) |
| `p07_cat_on_shelf.mp4` | `p07_red_fox.jpg` | `Replace the cat with the red fox from @image1.` | an animal |
