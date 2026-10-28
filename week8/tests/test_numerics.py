"""Independent fixtures: each test names the numerical defect it catches."""
import itertools
import numpy as np
import pytest


def test_story_split_preserves_order_and_drops_empty_stories():
    from src.data import split_stories
    train, valid = split_stories(' a <|endoftext|>\n<|endoftext|> b <|endoftext|> c', 2/3)
    assert train == ['a', 'b'] and valid == ['c']


def test_offsets_exclude_the_pair_itself():
    from src.embeddings import offset_cosines
    # Perpendicular offsets would correlate with a self-containing average.
    np.testing.assert_allclose(offset_cosines(np.eye(3)), 0, atol=1e-15)
    np.testing.assert_allclose(offset_cosines(np.ones((3, 2))), 1, atol=1e-15)


def test_count_model_add_k_uses_context_total_and_full_vocabulary():
    from src.ngrams import NGram
    m = NGram.fit(np.array([0, 1, 0, 1, 1]), 2, 3, 1.0)
    np.testing.assert_allclose(m.probabilities([0]), [1/5, 3/5, 1/5])
    np.testing.assert_allclose(m.probabilities([1]), [2/5, 2/5, 1/5])
    np.testing.assert_allclose(m.probabilities([2]), [1/3]*3)


def test_packed_ngrams_do_not_overflow_at_order_six():
    from src.ngrams import pack_windows
    assert pack_windows(np.array([511]*6), 6, 512).tolist() == [512**6-1]


def test_ngram_score_matches_explicit_chain_and_unseen_fraction():
    from src.ngrams import NGram
    m = NGram.fit(np.array([0, 1, 0, 1, 1]), 2, 3, 1.0)
    loss, unseen = m.score(np.array([0, 1, 2]))
    assert loss == pytest.approx(-np.log([3/5, 1/5]).mean())
    assert unseen == 0.5


def test_bigram_explicit_enumeration_normalizes_all_sequences():
    from src.ngrams import enumerate_mass
    p = np.array([[.25, .75], [.8, .2]])
    np.testing.assert_allclose(enumerate_mass(p, 0), [1, 1, 1], atol=1e-14)
    assert enumerate_mass(2*p, 0) == pytest.approx([2, 4, 8])


def test_bigram_sampler_conditions_on_immediate_predecessor():
    from src.ngrams import sample_chain
    assert sample_chain(np.array([[0., 1.], [1., 0.]]), 10, 0, 2).tolist() == [1,0]*5


def test_trigram_opening_uses_bigram_until_two_tokens_are_available():
    from src.ngrams import NGram,sample_ngram
    tokens=np.array([2,0,1,0,1,0,1])
    models={n:NGram.fit(tokens,n,3,k=0) for n in [1,2,3]}
    assert sample_ngram(models,3,6,2,1848).tolist()==[0,1,0,1,0,1]


def test_chisquare_pools_rare_categories_without_losing_mass():
    from src.sampling import chi_square
    stat, df, p = chi_square(np.array([50, 49, 1, 0]), np.array([.5, .48, .01, .01]))
    # Rare bin expected=2 joins the next-smallest bin, giving [50,50].
    assert stat == 0 and df == 1 and p == 1


@pytest.mark.parametrize('kind', ['outer', 'least_squares', 'relu1', 'relu2', 'relu3'])
def test_orthogonal_keys_recall_values_without_crosstalk(kind):
    from src.memory import recall
    k = np.eye(3); v = np.array([[1.,2.],[-3.,4.],[5.,6.]])
    out, own = recall(k, v, np.array([0,1,2]), kind)
    np.testing.assert_allclose(out, v, atol=1e-13)
    np.testing.assert_allclose(own, 1, atol=1e-13)


def test_least_squares_signal_is_projection_diagonal():
    from src.memory import recall
    k = np.ones((2, 1)); v = np.eye(2)
    out, own = recall(k, v, np.array([0,1]), 'least_squares')
    np.testing.assert_allclose(out, .5*np.ones((2,2)), atol=1e-13)
    np.testing.assert_allclose(own, [.5,.5], atol=1e-13)


def test_softmax_normalization_preserves_signal_to_interference():
    from src.memory import recall
    k=np.array([[1.,0.],[0.,1.]]); v=np.eye(2)
    out, own = recall(k,v,np.array([0]),'softmax',beta=2)
    np.testing.assert_allclose(out/own[:,None], [[1,np.exp(-2)]], atol=1e-13)


def test_softmax_context_memory_is_invariant_to_pair_permutation():
    from src.memory import recall
    k=np.array([[1.,0.],[0.,1.],[-1.,0.]])
    v=np.array([[2.,3.],[4.,-2.],[-1.,5.]])
    a,wa=recall(k,v,np.array([0]),'softmax',beta=2)
    order=np.array([1,2,0])
    b,wb=recall(k[order],v[order],np.array([2]),'softmax',beta=2)
    np.testing.assert_allclose(a,b,atol=1e-14)
    np.testing.assert_allclose(wa,wb,atol=1e-14)


def test_capacity_interpolates_in_log_space():
    from src.memory import capacity
    assert capacity(np.array([4,16]),np.array([.5,2])) == pytest.approx(8)


def test_top_p_keeps_threshold_crossing_token():
    from src.sampling import distribution
    np.testing.assert_allclose(distribution(np.log([.5,.3,.2]),top_p=.6),[.625,.375,0])


def test_top_k_and_min_p_renormalize_retained_mass():
    from src.sampling import distribution
    logits=np.log([.5,.3,.15,.05])
    np.testing.assert_allclose(distribution(logits,top_k=2),[.625,.375,0,0])
    np.testing.assert_allclose(distribution(logits,min_p=.3),[.5/.95,.3/.95,.15/.95,0])


def test_temperature_sharpens_before_truncation():
    from src.sampling import distribution
    np.testing.assert_allclose(distribution(np.log([.6,.4]),temperature=.5),[9/13,4/13])
    np.testing.assert_allclose(distribution([1,2,3],temperature=.0001),[0,0,1])


def test_scaling_optimum_obeys_budget_and_stationarity():
    from src.scaling import optimal_allocation, predict
    p=np.array([1.8,480,2100,.35,.37]); c=np.array([1e18,5.76e23,1e25])
    n,d=optimal_allocation(p,c)
    np.testing.assert_allclose(6*n*d,c,rtol=1e-13)
    np.testing.assert_allclose(p[3]*p[1]/n**p[3],p[4]*p[2]/d**p[4],rtol=1e-13)
    assert np.all(predict(p,n,d)<predict(p,2*n,d/2))


def test_huber_gradient_agrees_with_finite_differences():
    from src.scaling import objective
    theta=np.array([np.log(1.8),np.log(480),np.log(2100),.35,.37])
    ln=np.log([1e8,1e9,1e10]); ld=np.log([1e9,1e10,1e11]); y=np.log([4.5,3.2,2.5])
    value,grad=objective(theta,ln,ld,y)
    fd=[]
    for i in range(5):
        step=np.eye(5)[i]*1e-6
        fd.append((objective(theta+step,ln,ld,y)[0]-objective(theta-step,ln,ld,y)[0])/2e-6)
    np.testing.assert_allclose(grad,fd,rtol=1e-5,atol=1e-8)
