df <- read.csv("data_sorted_1.csv", header=T)
df[df$action == df$iaction,]
df$probability[df$action == df$iaction]

WhatIf -q latent_collision -e action\(cruise\),True -e curr_lane,True -e free_E,False -e free_NE,True -e free_NW,True -e free_SE,False -e free_SW,False -e free_W,False -e latent_collision,True  -i action\(change_to_left\),True -i action\(change_to_right\),False -i action\(cruise\),False -i action\(keep\),False -i action\(swerve_left\),False -i action\(swerve_right\),False ./cBNs/cBN_1.pl

Row: 1 Probability: 0.9992302129812992 Elapsed time 3.392606019973755

WhatIf -q latent_collision -e action\(cruise\),True -e curr_lane,True -e free_E,False -e free_NE,True -e free_NW,True -e free_SE,False -e free_SW,False -e free_W,False -e latent_collision,True  -i action\(change_to_left\),False -i action\(change_to_right\),True -i action\(cruise\),False -i action\(keep\),False -i action\(swerve_left\),False -i action\(swerve_right\),False ./cBNs/cBN_1.pl

Row: 2 Probability: 0.9992302129812991 Elapsed time 3.239720106124878

WhatIf -q latent_collision -e action\(cruise\),True -e curr_lane,True -e free_E,False -e free_NE,True -e free_NW,True -e free_SE,False -e free_SW,False -e free_W,False -e latent_collision,True  -i action\(change_to_left\),False -i action\(change_to_right\),False -i action\(cruise\),True -i action\(keep\),False -i action\(swerve_left\),False -i action\(swerve_right\),False ./cBNs/cBN_1.pl

Row: 3 Probability: 0.0021464420562651316 Elapsed time 3.293696641921997

WhatIf -q latent_collision -e action\(cruise\),True -e curr_lane,True -e free_E,False -e free_NE,True -e free_NW,True -e free_SE,False -e free_SW,False -e free_W,False -e latent_collision,True  -i action\(change_to_left\),False -i action\(change_to_right\),False -i action\(cruise\),False -i action\(keep\),True -i action\(swerve_left\),False -i action\(swerve_right\),False ./cBNs/cBN_1.pl

Row: 4 Probability: 0.5522085793988153 Elapsed time 3.27253794670105

WhatIf -q latent_collision -e action\(cruise\),True -e curr_lane,True -e free_E,False -e free_NE,True -e free_NW,True -e free_SE,False -e free_SW,False -e free_W,False -e latent_collision,True  -i action\(change_to_left\),False -i action\(change_to_right\),False -i action\(cruise\),False -i action\(keep\),False -i action\(swerve_left\),True -i action\(swerve_right\),False ./cBNs/cBN_1.pl

Row: 5 Probability: 0.5522085793988153 Elapsed time 3.2728517055511475

WhatIf -q latent_collision -e action\(cruise\),True -e curr_lane,True -e free_E,False -e free_NE,True -e free_NW,True -e free_SE,False -e free_SW,False -e free_W,False -e latent_collision,True  -i action\(change_to_left\),False -i action\(change_to_right\),False -i action\(cruise\),False -i action\(keep\),False -i action\(swerve_left\),False -i action\(swerve_right\),True ./cBNs/cBN_1.pl

Row: 6 Probability: 0.5522085793988153 Elapsed time 3.272110939025879


