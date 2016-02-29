#ifndef _NUMPY_FORMAT_H_
#define _NUMPY_FORMAT_H_


#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace numpy_format {
    template <typename Scalar>
    const char * dtype_for_scalar() {
        return "object";
    }


    // These specializations only work if the machine you are going to be
    // reading the file from has the same endianess
    template<>
    const char * dtype_for_scalar<int8_t>() { return "i1"; }
    template<>
    const char * dtype_for_scalar<int16_t>() { return "i2"; }
    template<>
    const char * dtype_for_scalar<int32_t>() { return "i4"; }
    template<>
    const char * dtype_for_scalar<int64_t>() { return "i8"; }
    template<>
    const char * dtype_for_scalar<uint8_t>() { return "u1"; }
    template<>
    const char * dtype_for_scalar<uint16_t>() { return "u2"; }
    template<>
    const char * dtype_for_scalar<uint32_t>() { return "u4"; }
    template<>
    const char * dtype_for_scalar<uint64_t>() { return "u8"; }
    template<>
    const char * dtype_for_scalar<float>() { return "f4"; }
    template<>
    const char * dtype_for_scalar<double>() { return "f8"; }


    bool is_big_endian() {
        union ByteOrder
        {
            int32_t i;
            uint8_t c[4];
        };
        ByteOrder b = {0x01234567};

        return b.c[0] == 0x01;
    }


    template <typename Scalar>
    void swap_endianess(Scalar * data, size_t N) {
        union D
        {
            Scalar v;
            uint8_t c[sizeof(Scalar)];
        };

        D d;
        for (size_t i=0; i<N; i++) {
            d.v = data[i];

            for (size_t j=0; j<sizeof(Scalar)/2; j++) {
                std::swap(d.c[j], d.c[sizeof(Scalar)-j-1]);
            }
        }
    }


    template <typename Scalar>
    class NumpyOutput
    {
        public:
            NumpyOutput(
                const Scalar *data,
                std::vector<size_t> shape,
                bool fortran
            ) : data_(data),
                shape_(shape),
                fortran_(fortran)
            {}

            template <int Rows, int Cols, int Options>
            NumpyOutput(const Eigen::Matrix<Scalar, Rows, Cols, Options> &matrix)
                : NumpyOutput(
                    matrix.data(),
                    std::vector<size_t>{
                        static_cast<size_t>(matrix.rows()),
                        static_cast<size_t>(matrix.cols())
                    },
                    ! Eigen::Matrix<Scalar, Rows, Cols, Options>::IsRowMajor
                )
            {}

            std::string dtype() const {
                return std::string(is_big_endian() ? ">" : "<") +
                       dtype_for_scalar<Scalar>();
            }

            const char * data() const { return reinterpret_cast<const char *>(data_); }
            const std::vector<size_t> & shape() const { return shape_; }
            bool fortran_contiguous() const { return fortran_; }

        private:
            /**
             * This operator implements the npy format for a block of NumpyOutput of type
             * Scalar.
             */
            friend std::ostream & operator<<(std::ostream &os, const NumpyOutput &data) {
                const uint8_t MAGIC_AND_VERSION[] = {
                    0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 0x01, 0x00
                };

                // We need to create the header for the data which is a python literal
                // string
                std::ostringstream header;
                header << "{'descr': '" << data.dtype() << "',"
                       << " 'fortran_order': " << (data.fortran_contiguous() ? "True" : "False") << ","
                       << " 'shape': (";
                for (auto d : data.shape()) {
                    header << d << ", ";
                }
                header << ")}";

                // Now we need to pad it with spaces until the total size of the magic +
                // version + header_len + header is divisible by 16
                while ((static_cast<size_t>(header.tellp()) + 11) % 16) {
                    header << " ";
                }
                header << "\n";

                // Now we need to write the magic and version
                os.write(reinterpret_cast<const char *>(MAGIC_AND_VERSION), 8);

                // Write the header length in little endian no matter what
                uint16_t header_len = header.tellp();
                if (!is_big_endian()) {
                    os.write(reinterpret_cast<const char *>(&header_len), 2);
                } else {
                    os.write(reinterpret_cast<const char *>(&header_len) + 1, 1);
                    os.write(reinterpret_cast<const char *>(&header_len), 1);
                }

                // Write the header
                os << header.str();

                // Finally write the data
                size_t N = 1;
                for (auto d : data.shape()) {
                    N *= d;
                }
                os.write(data.data(), N*sizeof(Scalar));

                return os;
            }

            const Scalar * data_;
            std::vector<size_t> shape_;
            bool fortran_;
    };


    template <typename Scalar>
    class NumpyInput
    {
        public:
            NumpyInput() {}

            const bool fortran_contiguous() const { return fortran_; }
            const Scalar * data() const { return data_.data(); }
            const std::vector<size_t> & shape() const { return shape_; }

            template <int Rows, int Cols, int Options>
            operator Eigen::Matrix<Scalar, Rows, Cols, Options>() const {
                // get the needed rows, cols
                int rows = shape_[0];
                int cols = 1;
                for (size_t i=1; i<shape_.size(); i++) {
                    cols *= shape_[i];
                }

                Eigen::Matrix<Scalar, Rows, Cols, Options> matrix(
                    rows,
                    cols
                );

                // copy the data by hand for simplicity
                for (int i=0; i<cols; i++) {
                    for (int j=0; j<rows; j++) {
                        int idx = (fortran_) ? i*rows + j : j*cols + i;

                        matrix(j, i) = data_[idx];
                    }
                }

                // now return the object and let c++11 move semantics make it a
                // cost free return by value
                return matrix;
            }

        private:
            friend std::istream & operator>>(std::istream &is, NumpyInput &data) {
                // reset the NumpyInput instance
                data.data_.clear();
                data.shape_.clear();

                // read the magic and the version and assert that it is compatible
                char MAGIC_AND_VERSION[8];
                is.read(MAGIC_AND_VERSION, 8);
                if (MAGIC_AND_VERSION[6] > 1) {
                    throw std::runtime_error(
                        "Only version 1 of the numpy format is supported"
                    );
                }

                // if the file is empty (aka we read nothing) just throw a
                // runtime error
                if (is.gcount() == 0) {
                    throw std::runtime_error(
                        "The file is empty and cannot be read"
                    );
                }

                // read the header len
                uint16_t header_len;
                is.read(reinterpret_cast<char *>(&header_len), 2);
                if (is_big_endian()) {
                    swap_endianess(&header_len, 1);
                }

                // read the header
                std::vector<char> buffer(header_len+1);
                is.read(&buffer[0], header_len);
                buffer[header_len] = 0;
                std::string header(&buffer[0]);

                // we can parse the header efficiently using the fact that the
                // specification requires the dictionary to be passed by
                // pprint.pformat()

                // parse dtype info
                std::string dtype = header.substr(11, 3);
                bool endianness = dtype[0] == '>';
                if (dtype.substr(1) != dtype_for_scalar<Scalar>()) {
                    throw std::runtime_error(
                        std::string() + 
                        "The type of the array is not the " +
                        "one requested: " + dtype.substr(1) +
                        " != " + dtype_for_scalar<Scalar>()
                    );
                }

                // parse contiguity type
                data.fortran_ = header[34] == 'T';

                // parse shape
                std::string shape = header.substr(
                    header.find_last_of('(')+1,
                    header.find_last_of(')')
                );
                try {
                    while (true) {
                        size_t processed;
                        data.shape_.push_back(std::stoi(shape, &processed));

                        // +2 to account for the comma and the space
                        shape = shape.substr(processed + 2);
                    }
                } catch (const std::invalid_argument&) {
                    // that's ok it means we finished parsing the tuple
                }

                // compute the total size of the data
                int N = 1;
                for (auto c : data.shape_) {
                    N *= c;
                }

                // read the data
                data.data_.resize(N);
                is.read(reinterpret_cast<char *>(&data.data_[0]), N*sizeof(Scalar));

                // fix the endianess
                if (endianness != is_big_endian()) {
                    swap_endianess(&data.data_[0], N);
                }

                return is;
            }

            std::vector<Scalar> data_;
            std::vector<size_t> shape_;
            bool fortran_;
    };


    /**
     * Saves data into a stream  
     */
    template <typename Scalar, int Rows, int Cols, int Options>
    void save(std::ostream &out_stream, const Eigen::Matrix<Scalar, Rows, Cols, Options> &matrix) {
        out_stream << NumpyOutput<Scalar>(matrix);
    }


    /**
     * Saves data into a file provided as input
     */
    template <typename Scalar, int Rows, int Cols, int Options>
    void save(std::string save_path, const Eigen::Matrix<Scalar, Rows, Cols, Options> &matrix) {
        std::fstream out_stream(
            save_path,
            std::ios::out | std::ios::binary
        );

        save(out_stream, matrix);
    }


    /**
     * Loads data from a stream
     */
    template <
        typename Scalar,
        int Rows=Eigen::Dynamic,
        int Cols=Eigen::Dynamic,
        int Options=Eigen::ColMajor | Eigen::AutoAlign
    >
    Eigen::Matrix<Scalar, Rows, Cols, Options> load(std::istream &in_stream) {
        NumpyInput<Scalar> ni;
        in_stream >> ni;

        return ni;
    }


    /**
     * Loads data from a file provided as input
     */
    template <
        typename Scalar,
        int Rows=Eigen::Dynamic,
        int Cols=Eigen::Dynamic,
        int Options=Eigen::ColMajor | Eigen::AutoAlign
    >
    Eigen::Matrix<Scalar, Rows, Cols, Options> load(std::string data_path) {
        std::fstream in_stream(
            data_path,
            std::ios::in | std::ios::binary
        );
        
        return load<Scalar>(in_stream);
    }

} // namespace numpy_format 

#endif
