"""Active model shared by training and the dashboard."""
ACTIVE_EXPERIMENT = 'skillcorner_20_compact'
ACTIVE_EXCLUDE = '(?:^(own_n|opp_n|n_tracked|ball_known|ball_seen|ball_seen_m1\\.0|opp_touches_before)$)|(?:^(own_cx_m1\\.0|opp_cx_m1\\.0|own_cx_m3\\.0|opp_cx_m3\\.0)$)'
ACTIVE_DEAD_EXCLUDE = '^team1\\.(own_n2|opp_n2|own_touches_before)$'
ACTIVE_POSSESSION_EXCLUDE = '^shared\\.ball_since_seen_s$'


GB_PARAMS = dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                 min_samples_leaf=400, l2_regularization=1.0,
                 early_stopping=False, random_state=0)
DECODER = dict(a_dead=0.75, a_team=2.0, switch=64., stop=40., restart=40., min_dead=25)
