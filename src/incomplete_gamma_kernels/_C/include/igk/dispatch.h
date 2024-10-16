#pragma once

#include <c10/util/Exception.h>

namespace igk
{

#define IGK_DETAIL_STRINGIFY_IMPL(x) #x
#define IGK_STRINGIFY(x) IGK_DETAIL_STRINGIFY_IMPL(x)

template <typename T>
struct dispatch_traits;

#define IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION, ...)                                                       \
    if (VALUE == dispatch_traits<decltype(OPTION)>::PROPERTY)                                                          \
    {                                                                                                                  \
        [[maybe_unused]] auto functor = OPTION;                                                                        \
        __VA_ARGS__();                                                                                                 \
        break;                                                                                                         \
    }

#define IGK_DETAIL_DISPATCH_FALLBACK(VALUE)                                                                            \
    {                                                                                                                  \
        auto msg = std::string("Unsupported ") + IGK_STRINGIFY(VALUE) + " \"" + VALUE + "\".";                         \
        TORCH_CHECK(false, msg);                                                                                       \
    }

#define IGK_DISPATCH_FUNCTORS_BY_STRING_1(VALUE, PROPERTY, OPTION1, ...)                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION1, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_FALLBACK(VALUE)                                                                            \
    } while (0)

#define IGK_DISPATCH_FUNCTORS_BY_STRING_2(VALUE, PROPERTY, OPTION1, OPTION2, ...)                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION1, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION2, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_FALLBACK(VALUE)                                                                            \
    } while (0)

#define IGK_DISPATCH_FUNCTORS_BY_STRING_3(VALUE, PROPERTY, OPTION1, OPTION2, OPTION3, ...)                             \
    do                                                                                                                 \
    {                                                                                                                  \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION1, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION2, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION3, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_FALLBACK(VALUE)                                                                            \
    } while (0)

#define IGK_DISPATCH_FUNCTORS_BY_STRING_4(VALUE, PROPERTY, OPTION1, OPTION2, OPTION3, OPTION4, ...)                    \
    do                                                                                                                 \
    {                                                                                                                  \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION1, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION2, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION3, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION4, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_FALLBACK(VALUE)                                                                            \
    } while (0)

#define IGK_DISPATCH_FUNCTORS_BY_STRING_5(VALUE, PROPERTY, OPTION1, OPTION2, OPTION3, OPTION4, OPTION5, ...)           \
    do                                                                                                                 \
    {                                                                                                                  \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION1, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION2, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION3, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION4, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION5, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_FALLBACK(VALUE)                                                                            \
    } while (0)

#define IGK_DISPATCH_FUNCTORS_BY_STRING_6(VALUE, PROPERTY, OPTION1, OPTION2, OPTION3, OPTION4, OPTION5, OPTION6, ...)  \
    do                                                                                                                 \
    {                                                                                                                  \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION1, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION2, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION3, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION4, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION5, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION6, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_FALLBACK(VALUE)                                                                            \
    } while (0)

#define IGK_DISPATCH_FUNCTORS_BY_STRING_7(VALUE,                                                                       \
                                          PROPERTY,                                                                    \
                                          OPTION1,                                                                     \
                                          OPTION2,                                                                     \
                                          OPTION3,                                                                     \
                                          OPTION4,                                                                     \
                                          OPTION5,                                                                     \
                                          OPTION6,                                                                     \
                                          OPTION7,                                                                     \
                                          ...)                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION1, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION2, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION3, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION4, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION5, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION6, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION7, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_FALLBACK(VALUE)                                                                            \
    } while (0)

#define IGK_DISPATCH_FUNCTORS_BY_STRING_8(VALUE,                                                                       \
                                          PROPERTY,                                                                    \
                                          OPTION1,                                                                     \
                                          OPTION2,                                                                     \
                                          OPTION3,                                                                     \
                                          OPTION4,                                                                     \
                                          OPTION5,                                                                     \
                                          OPTION6,                                                                     \
                                          OPTION7,                                                                     \
                                          OPTION8,                                                                     \
                                          ...)                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION1, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION2, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION3, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION4, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION5, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION6, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION7, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION8, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_FALLBACK(VALUE)                                                                            \
    } while (0)

#define IGK_DISPATCH_FUNCTORS_BY_STRING_9(VALUE,                                                                       \
                                          PROPERTY,                                                                    \
                                          OPTION1,                                                                     \
                                          OPTION2,                                                                     \
                                          OPTION3,                                                                     \
                                          OPTION4,                                                                     \
                                          OPTION5,                                                                     \
                                          OPTION6,                                                                     \
                                          OPTION7,                                                                     \
                                          OPTION8,                                                                     \
                                          OPTION9,                                                                     \
                                          ...)                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION1, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION2, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION3, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION4, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION5, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION6, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION7, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION8, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_OPTION(VALUE, PROPERTY, OPTION9, __VA_ARGS__)                                              \
        IGK_DETAIL_DISPATCH_FALLBACK(VALUE)                                                                            \
    } while (0)

} // namespace igk
