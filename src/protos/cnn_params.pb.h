// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: cnn_params.proto

#ifndef PROTOBUF_cnn_5fparams_2eproto__INCLUDED
#define PROTOBUF_cnn_5fparams_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3004000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3004000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
namespace CnnProto {
class Bias;
class BiasDefaultTypeInternal;
extern BiasDefaultTypeInternal _Bias_default_instance_;
class Params;
class ParamsDefaultTypeInternal;
extern ParamsDefaultTypeInternal _Params_default_instance_;
class Weight;
class WeightDefaultTypeInternal;
extern WeightDefaultTypeInternal _Weight_default_instance_;
}  // namespace CnnProto

namespace CnnProto {

namespace protobuf_cnn_5fparams_2eproto {
// Internal implementation detail -- do not call these.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[];
  static const ::google::protobuf::uint32 offsets[];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static void InitDefaultsImpl();
};
void AddDescriptors();
void InitDefaults();
}  // namespace protobuf_cnn_5fparams_2eproto

// ===================================================================

class Weight : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:CnnProto.Weight) */ {
 public:
  Weight();
  virtual ~Weight();

  Weight(const Weight& from);

  inline Weight& operator=(const Weight& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  Weight(Weight&& from) noexcept
    : Weight() {
    *this = ::std::move(from);
  }

  inline Weight& operator=(Weight&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const Weight& default_instance();

  static inline const Weight* internal_default_instance() {
    return reinterpret_cast<const Weight*>(
               &_Weight_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    0;

  void Swap(Weight* other);
  friend void swap(Weight& a, Weight& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline Weight* New() const PROTOBUF_FINAL { return New(NULL); }

  Weight* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const Weight& from);
  void MergeFrom(const Weight& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(Weight* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated double w = 1 [packed = true];
  int w_size() const;
  void clear_w();
  static const int kWFieldNumber = 1;
  double w(int index) const;
  void set_w(int index, double value);
  void add_w(double value);
  const ::google::protobuf::RepeatedField< double >&
      w() const;
  ::google::protobuf::RepeatedField< double >*
      mutable_w();

  // @@protoc_insertion_point(class_scope:CnnProto.Weight)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::RepeatedField< double > w_;
  mutable int _w_cached_byte_size_;
  mutable int _cached_size_;
  friend struct protobuf_cnn_5fparams_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class Bias : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:CnnProto.Bias) */ {
 public:
  Bias();
  virtual ~Bias();

  Bias(const Bias& from);

  inline Bias& operator=(const Bias& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  Bias(Bias&& from) noexcept
    : Bias() {
    *this = ::std::move(from);
  }

  inline Bias& operator=(Bias&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const Bias& default_instance();

  static inline const Bias* internal_default_instance() {
    return reinterpret_cast<const Bias*>(
               &_Bias_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    1;

  void Swap(Bias* other);
  friend void swap(Bias& a, Bias& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline Bias* New() const PROTOBUF_FINAL { return New(NULL); }

  Bias* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const Bias& from);
  void MergeFrom(const Bias& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(Bias* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated double b = 1 [packed = true];
  int b_size() const;
  void clear_b();
  static const int kBFieldNumber = 1;
  double b(int index) const;
  void set_b(int index, double value);
  void add_b(double value);
  const ::google::protobuf::RepeatedField< double >&
      b() const;
  ::google::protobuf::RepeatedField< double >*
      mutable_b();

  // @@protoc_insertion_point(class_scope:CnnProto.Bias)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::RepeatedField< double > b_;
  mutable int _b_cached_byte_size_;
  mutable int _cached_size_;
  friend struct protobuf_cnn_5fparams_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class Params : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:CnnProto.Params) */ {
 public:
  Params();
  virtual ~Params();

  Params(const Params& from);

  inline Params& operator=(const Params& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  Params(Params&& from) noexcept
    : Params() {
    *this = ::std::move(from);
  }

  inline Params& operator=(Params&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const Params& default_instance();

  static inline const Params* internal_default_instance() {
    return reinterpret_cast<const Params*>(
               &_Params_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    2;

  void Swap(Params* other);
  friend void swap(Params& a, Params& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline Params* New() const PROTOBUF_FINAL { return New(NULL); }

  Params* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const Params& from);
  void MergeFrom(const Params& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(Params* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated .CnnProto.Weight weights = 1;
  int weights_size() const;
  void clear_weights();
  static const int kWeightsFieldNumber = 1;
  const ::CnnProto::Weight& weights(int index) const;
  ::CnnProto::Weight* mutable_weights(int index);
  ::CnnProto::Weight* add_weights();
  ::google::protobuf::RepeatedPtrField< ::CnnProto::Weight >*
      mutable_weights();
  const ::google::protobuf::RepeatedPtrField< ::CnnProto::Weight >&
      weights() const;

  // repeated .CnnProto.Bias biases = 2;
  int biases_size() const;
  void clear_biases();
  static const int kBiasesFieldNumber = 2;
  const ::CnnProto::Bias& biases(int index) const;
  ::CnnProto::Bias* mutable_biases(int index);
  ::CnnProto::Bias* add_biases();
  ::google::protobuf::RepeatedPtrField< ::CnnProto::Bias >*
      mutable_biases();
  const ::google::protobuf::RepeatedPtrField< ::CnnProto::Bias >&
      biases() const;

  // @@protoc_insertion_point(class_scope:CnnProto.Params)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::RepeatedPtrField< ::CnnProto::Weight > weights_;
  ::google::protobuf::RepeatedPtrField< ::CnnProto::Bias > biases_;
  mutable int _cached_size_;
  friend struct protobuf_cnn_5fparams_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// Weight

// repeated double w = 1 [packed = true];
inline int Weight::w_size() const {
  return w_.size();
}
inline void Weight::clear_w() {
  w_.Clear();
}
inline double Weight::w(int index) const {
  // @@protoc_insertion_point(field_get:CnnProto.Weight.w)
  return w_.Get(index);
}
inline void Weight::set_w(int index, double value) {
  w_.Set(index, value);
  // @@protoc_insertion_point(field_set:CnnProto.Weight.w)
}
inline void Weight::add_w(double value) {
  w_.Add(value);
  // @@protoc_insertion_point(field_add:CnnProto.Weight.w)
}
inline const ::google::protobuf::RepeatedField< double >&
Weight::w() const {
  // @@protoc_insertion_point(field_list:CnnProto.Weight.w)
  return w_;
}
inline ::google::protobuf::RepeatedField< double >*
Weight::mutable_w() {
  // @@protoc_insertion_point(field_mutable_list:CnnProto.Weight.w)
  return &w_;
}

// -------------------------------------------------------------------

// Bias

// repeated double b = 1 [packed = true];
inline int Bias::b_size() const {
  return b_.size();
}
inline void Bias::clear_b() {
  b_.Clear();
}
inline double Bias::b(int index) const {
  // @@protoc_insertion_point(field_get:CnnProto.Bias.b)
  return b_.Get(index);
}
inline void Bias::set_b(int index, double value) {
  b_.Set(index, value);
  // @@protoc_insertion_point(field_set:CnnProto.Bias.b)
}
inline void Bias::add_b(double value) {
  b_.Add(value);
  // @@protoc_insertion_point(field_add:CnnProto.Bias.b)
}
inline const ::google::protobuf::RepeatedField< double >&
Bias::b() const {
  // @@protoc_insertion_point(field_list:CnnProto.Bias.b)
  return b_;
}
inline ::google::protobuf::RepeatedField< double >*
Bias::mutable_b() {
  // @@protoc_insertion_point(field_mutable_list:CnnProto.Bias.b)
  return &b_;
}

// -------------------------------------------------------------------

// Params

// repeated .CnnProto.Weight weights = 1;
inline int Params::weights_size() const {
  return weights_.size();
}
inline void Params::clear_weights() {
  weights_.Clear();
}
inline const ::CnnProto::Weight& Params::weights(int index) const {
  // @@protoc_insertion_point(field_get:CnnProto.Params.weights)
  return weights_.Get(index);
}
inline ::CnnProto::Weight* Params::mutable_weights(int index) {
  // @@protoc_insertion_point(field_mutable:CnnProto.Params.weights)
  return weights_.Mutable(index);
}
inline ::CnnProto::Weight* Params::add_weights() {
  // @@protoc_insertion_point(field_add:CnnProto.Params.weights)
  return weights_.Add();
}
inline ::google::protobuf::RepeatedPtrField< ::CnnProto::Weight >*
Params::mutable_weights() {
  // @@protoc_insertion_point(field_mutable_list:CnnProto.Params.weights)
  return &weights_;
}
inline const ::google::protobuf::RepeatedPtrField< ::CnnProto::Weight >&
Params::weights() const {
  // @@protoc_insertion_point(field_list:CnnProto.Params.weights)
  return weights_;
}

// repeated .CnnProto.Bias biases = 2;
inline int Params::biases_size() const {
  return biases_.size();
}
inline void Params::clear_biases() {
  biases_.Clear();
}
inline const ::CnnProto::Bias& Params::biases(int index) const {
  // @@protoc_insertion_point(field_get:CnnProto.Params.biases)
  return biases_.Get(index);
}
inline ::CnnProto::Bias* Params::mutable_biases(int index) {
  // @@protoc_insertion_point(field_mutable:CnnProto.Params.biases)
  return biases_.Mutable(index);
}
inline ::CnnProto::Bias* Params::add_biases() {
  // @@protoc_insertion_point(field_add:CnnProto.Params.biases)
  return biases_.Add();
}
inline ::google::protobuf::RepeatedPtrField< ::CnnProto::Bias >*
Params::mutable_biases() {
  // @@protoc_insertion_point(field_mutable_list:CnnProto.Params.biases)
  return &biases_;
}
inline const ::google::protobuf::RepeatedPtrField< ::CnnProto::Bias >&
Params::biases() const {
  // @@protoc_insertion_point(field_list:CnnProto.Params.biases)
  return biases_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS
// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)


}  // namespace CnnProto

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_cnn_5fparams_2eproto__INCLUDED
