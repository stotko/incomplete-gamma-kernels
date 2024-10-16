#pragma once

#include <c10/macros/Macros.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>

namespace igk
{

template <typename Incrementable,
          typename System = thrust::use_default,
          typename Traversal = thrust::use_default,
          typename Difference = thrust::use_default>
class arange_iterator
  : public thrust::iterator_adaptor<arange_iterator<Incrementable, System, Traversal, Difference>,
                                    thrust::counting_iterator<Incrementable, System, Traversal, Difference>>
{
public:
    using super_t =
            typename thrust::iterator_adaptor<arange_iterator<Incrementable, System, Traversal, Difference>,
                                              thrust::counting_iterator<Incrementable, System, Traversal, Difference>>;

    friend class thrust::iterator_core_access;

    inline C10_HOST_DEVICE explicit arange_iterator(const Incrementable& start,
                                                    const Incrementable& step = Incrementable{ 1 })
      : super_t(thrust::counting_iterator<Incrementable, System, Traversal, Difference>{ 0 })
      , start_(start)
      , step_(step)
    {
    }

private:
    inline C10_HOST_DEVICE typename super_t::reference
    dereference() const
    {
        return start_ + *(this->base()) * step_;
    }

    Incrementable start_;
    Incrementable step_;
};

} // namespace igk
