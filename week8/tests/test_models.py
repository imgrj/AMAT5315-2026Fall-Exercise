import numpy as np
import pytest
import jax
import jax.numpy as jnp


def test_mlp_cross_entropy_equals_numpy_bigram_table():
    from src.models import init_mlp, mlp, cross_entropy
    p=init_mlp(jax.random.key(1),vocab=5,dim=3,hidden=7)
    x=jnp.array([0,1,3]); y=jnp.array([1,4,2]); logits=np.asarray(mlp(p,x))
    probs=np.exp(logits-logits.max(-1,keepdims=True)); probs/=probs.sum(-1,keepdims=True)
    assert float(cross_entropy(mlp(p,x),y)) == pytest.approx(-np.log(probs[np.arange(3),y]).mean(),abs=1e-6)
    grad=jax.grad(lambda z:cross_entropy(mlp(z,x),y))(p)
    assert all(np.all(np.isfinite(v)) for v in jax.tree.leaves(grad))
    # Independent NumPy forward and chain-rule backward pass.
    e=np.asarray(p['embedding'])[np.asarray(x)]
    h=np.tanh(e@np.asarray(p['hidden']['w'])+np.asarray(p['hidden']['b']))
    z=h@np.asarray(p['readout']['w'])+np.asarray(p['readout']['b'])
    np.testing.assert_allclose(logits,z,atol=1e-7)
    expected=np.exp(z-z.max(-1,keepdims=True));expected/=expected.sum(-1,keepdims=True)
    expected[np.arange(3),np.asarray(y)]-=1;expected/=3
    dh=(expected@np.asarray(p['readout']['w']).T)*(1-h*h)
    np.testing.assert_allclose(grad['readout']['w'],h.T@expected,atol=1e-7)
    np.testing.assert_allclose(grad['hidden']['w'],e.T@dh,atol=1e-7)
    np.testing.assert_allclose(grad['hidden']['b'],dh.sum(0),atol=1e-7)


def test_attention_matches_hand_computed_two_token_average():
    from src.models import attention
    q=jnp.zeros((1,2,1,2)); k=q; v=jnp.array([[[[2.,4.]],[[6.,8.]]]])
    out,w=attention(q,k,v)
    np.testing.assert_allclose(out,[[[[2,4]],[[4,6]]]],atol=1e-7)
    np.testing.assert_allclose(w,[[[[1,0],[.5,.5]]]],atol=1e-7)


def test_attention_scales_by_head_dimension():
    from src.models import attention
    q=jnp.array([[[[2.,0.]],[[0.,2.]]]]); v=jnp.array([[[[1.,0.]],[[0.,1.]]]])
    _,w=attention(q,q,v)
    a=np.exp(4/np.sqrt(2)); np.testing.assert_allclose(w[0,0,1],[1/(1+a),a/(1+a)],rtol=1e-6)


def test_transformer_cannot_read_future_tokens():
    from src.models import init_transformer, transformer
    p=init_transformer(jax.random.key(4),vocab=9,dim=8,heads=2,layers=2,context=6)
    a,w=transformer(p,jnp.array([[1,2,3,4,5,6]]),heads=2,return_attention=True)
    b=transformer(p,jnp.array([[1,2,3,7,8,0]]),heads=2)
    np.testing.assert_allclose(a[:,:3],b[:,:3],atol=1e-7)
    for weights in w:
        assert np.all(np.asarray(weights)[...,np.triu_indices(6,1)[0],np.triu_indices(6,1)[1]]==0)
        np.testing.assert_allclose(weights.sum(-1),1,atol=2e-7)


def test_transformer_parameter_count_includes_all_biases():
    from src.models import init_transformer
    p=init_transformer(jax.random.key(0))
    # Vd+Td + L*(12d^2+13d) +2d +dV+V, from Fig. 14.
    assert sum(a.size for a in jax.tree.leaves(p)) == 328832


def test_learning_rate_has_warmup_peak_and_one_tenth_final():
    from src.training import learning_rate
    assert float(learning_rate(0,6000,300,.003)) == 0
    assert float(learning_rate(300,6000,300,.003)) == pytest.approx(.003)
    assert float(learning_rate(5999,6000,300,.003)) == pytest.approx(.0003)


@pytest.mark.parametrize('settings,expected', [
    ({'top_p':.6},[.625,.375,0]),
    ({'top_k':1},[1,0,0]),
    ({'min_p':.5},[.625,.375,0]),
])
def test_jax_sampler_retains_and_renormalizes_correct_tokens(settings,expected):
    from src.generation import probabilities
    np.testing.assert_allclose(probabilities(jnp.log(jnp.array([.5,.3,.2])),**settings),expected,atol=1e-7)


def test_generation_surprise_and_entropy_follow_original_model():
    from src.generation import generate
    from src.models import init_transformer
    p=init_transformer(jax.random.key(9),vocab=5,dim=8,heads=2,layers=1,context=4)
    # Zero readout makes every prediction exactly uniform, including sliding windows.
    p['readout']['w']=jnp.zeros_like(p['readout']['w']);p['readout']['b']=jnp.zeros(5)
    tokens,surprise,entropy=generate(p,seed=1,batch=3,length=9,eos=2,heads=2)
    assert tokens.shape==(3,9)
    np.testing.assert_allclose(surprise,np.log(5),atol=1e-6)
    np.testing.assert_allclose(entropy,np.log(5),atol=1e-6)


def test_shared_draw_helper_obeys_zero_probability_tokens():
    from src.generation import draw_tokens
    samples=draw_tokens(jax.random.key(11),jnp.array([0.,1.,0.]),shape=(1000,))
    assert np.all(np.asarray(samples)==1)
