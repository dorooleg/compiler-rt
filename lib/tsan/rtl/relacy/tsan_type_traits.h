#ifndef TSAN_TYPE_TRAITS_H
#define TSAN_TYPE_TRAITS_H

namespace __tsan {
namespace __relacy {

template<bool B, class T = void>
struct enable_if {
};

template<class T>
struct enable_if<true, T> {
  typedef T type;
};

template< class T > struct remove_const          { typedef T type; };
template< class T > struct remove_const<const T> { typedef T type; };

template< class T > struct remove_volatile             { typedef T type; };
template< class T > struct remove_volatile<volatile T> { typedef T type; };

template<class T>
struct remove_cv {
  typedef typename remove_volatile<typename remove_const<T>::type>::type type;
};

template<class T>
struct is_pointer_helper {
  constexpr static bool value = false;
};

template<class T>
struct is_pointer_helper<T *> {
  constexpr static bool value = true;
};

template<class T>
struct is_pointer : is_pointer_helper<typename remove_cv<T>::type> {
};

}
}

#endif //TSAN_TYPE_TRAITS_H
