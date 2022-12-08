import numpy as np
import plotly.express as px


def ldc_mat(reps_A: np.ndarray, reps_B: np.ndarray) -> np.ndarray:
    """
    Produces a cross-validated distance matrix using LDC distance, as described in https://pdf.sciencedirectassets.com/272508/1-s2.0-S1053811916X00110/1-s2.0-S1053811915011258/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEF0aCXVzLWVhc3QtMSJHMEUCIQCdRFbgXY5qKxtGuAsrHVVY%2BmgdmYCjF0kzmkiQOuJOOgIgafwSp9Kxf1IY%2Bm3Nkf6BcIiZuei2INMoPAdQsASPYSoqzAQIdhAFGgwwNTkwMDM1NDY4NjUiDOy%2BEkR%2F9IXY%2FQFyhiqpBFKmHYpPAM2XfkrRY9SDK7rwchau8xn0psbdYtw3%2F3YgtPJNKw58yQjUAKHZ535syZF%2FwpDSF90ueALgCjdGETUTnBwOU%2Fob2PiHf5tir%2FRxLnpel4BiTrPmVywELtVYoFIPfjUwVb6ORuceAJzdIpwwpOncQN0ly6HG7YS4vW1XxK697F4aNnk%2BCllnLijSb3mdgD1Y28Reopr4XPqVMedfMuiJoQtjtSQRUShJ%2F5O2GXykdiCtR9C4udi5uYOOMe2QZJJu%2FEXIkBpdER6m%2FvpVfNgBY%2B1yXb8pgkglaB1suqoByc%2F9lEviCJLZ1cQgcMLdZtgY9Xe7sbt8yM2bK51Fs22Psa6GVYZthKts%2BRE94MD7nfFUu1Ml1k0tu2SNGKMDXzfgiswh5PQU1Cyrlwsl4%2FzkRzD6h2eQOTrO4drcgxanj2g%2Bw2lHh9PghDjQwBPm99J%2FBhvzQZ6d4BbCMz6IZr8ZlW8fhMKzSHCuB9%2F4EAsdg9iUqO63QmSXQ%2FJSsF%2BPH1f3uOf5loNCQWLjHDa3W9jn81YcB2gKing0fxCSgjBYU8P6pd6siUWUonpwufAvqU%2Ba%2BnWsmbudk31tvOI1RmWn9miuyFvh5I8pycSIP3ezAhN9pFN%2B0MGM8DY5T7L29KY3CDz1myqA%2F%2Btjl9cq41JJCy3iC9oQp0NfXZR0kAYnTLCd0E%2BOIuNPx4z8UKi1bEoPXgZ6xnCC7wktF3ru5PrssfxfxaUw6qCdnAY6qQHHFb7nG%2FcAStLLDZ8H7GiMetpWPM%2FktEQIvv9Lwp%2B8UQ%2B%2Fa7DEP5qsTKFX%2FpfZTQHriGzHBlhtF0sxAdV76r3QctN5VmYYLmAG5ExYEXLzehG%2BHBzsUKzxwzogLEgb67HwTbWUcQlx%2BaMrBu5iBSQKJT4eqa62SyICiyGs%2BQSFn7nvXfooe40xXS85yMAaO57cKNeXn2XopyzShDlGladdOzsdHRLsrtx2&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20221130T133658Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY7INQXXIH%2F20221130%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=670c9b03547a13fd497a366621926c99aa6ed3605cff78214463a1640594d750&hash=b5457f365e37d54bb78d650ef4dd6d792a2207d6c2c11d3face535113a2d8c95&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1053811915011258&tid=spdf-25c5656b-fbc1-4c44-8db2-b0b6b0652e4e&sid=92e55c371b685944070bdef4cf5a0c4202b2gxrqb&type=client&ua=555050500859570d53&rr=7723f99c3fb47697
    """
    cov = np.cov(reps_A.T)
    
    # calculating pairwise b_k - b_j for all k and j
    num_chars = reps_A.shape[0]
    dim = reps_A.shape[1]

    delta_A = reps_A.repeat(num_chars, axis=0).reshape(num_chars, num_chars, dim) - reps_A
    delta_B = reps_B.repeat(num_chars, axis=0).reshape(num_chars, num_chars, dim) - reps_B
    delta_A = delta_A.reshape(1, num_chars ** 2, dim).squeeze(0)
    delta_B = delta_B.reshape(1, num_chars ** 2, dim).squeeze(0)

    # calculating all ldc distances
    cov_inv = np.eye(cov.shape[0], cov.shape[1])#np.linalg.inv(cov)
    ldc_dists = delta_A @ cov_inv @ delta_B.T

    # return just the diagonal (i.e. distance between b_k - b_j validated by the second phase)
    dist_mat = np.diagonal(ldc_dists).reshape(num_chars, num_chars)
    if not np.allclose(dist_mat, dist_mat.T):
        px.imshow(ldc_dists).show()
        px.imshow(dist_mat).show()
    
    return dist_mat