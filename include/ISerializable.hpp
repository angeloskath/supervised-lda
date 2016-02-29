#ifndef _I_SERIALIZABLE_HPP_
#define _I_SERIALIZABLE_HPP_


#include <vector>


/**
 * ISerializable aims to facilitate serialization of internal classes which
 * will be recreated from the serialized data using an IInternalsFactory.
 */
template <typename Scalar>
class ISerializable
{
    public:
        /**
         * A unique id identifying a class within a context. For instance a
         * unique id among Initialization strategies can be also used by an
         * expectation step.
         */
        virtual int get_id() = 0;
        /**
         * A set of parameters from which the object can be recreated.
         */
        virtual std::vector<Scalar> get_parameters() = 0;
        /**
         * Recreate the internal state from the given parameters.
         */
        virtual void set_parameters(std::vector<Scalar> parameters) = 0;
};


#endif  // _I_SERIALIZABLE_HPP_
