#include <pybind11/pybind11.h>
#include <cstdint>
#include <string>
#include <stdexcept>
namespace py = pybind11;

// Small deterministic CPU side stream. The neural/rANS alphabets stay intact.
// Rare signed int16 q -> u=2*(abs(q)-4)+(q<0), order0 Exp-Golomb(u).
py::bytes encode(const py::bytes& raw_object) {
    std::string raw=raw_object;
    if(raw.size()%2) throw std::invalid_argument("unaligned int16 bypass");
    std::string out; out.reserve(raw.size()); size_t pos=0;
    auto bit=[&](unsigned value) {
        if(!(pos%8)) out.push_back(0);
        if(value) out.back()=static_cast<char>(static_cast<unsigned char>(out.back()) | (1u<<(7-pos%8)));
        ++pos;
    };
    for(size_t i=0;i<raw.size();i+=2) {
        uint16_t u16=static_cast<unsigned char>(raw[i]) | (static_cast<uint16_t>(static_cast<unsigned char>(raw[i+1]))<<8);
        int q=u16>=32768 ? static_cast<int>(u16)-65536 : static_cast<int>(u16);
        int mag=q<0 ? -q : q;
        if(mag<4) throw std::invalid_argument("nonrare bypass symbol");
        unsigned code=2u*(mag-4)+(q<0)+1, k=0;
        for(unsigned v=code;v>>=1;) ++k;
        for(unsigned j=0;j<k;++j) bit(0);
        for(int j=static_cast<int>(k);j>=0;--j) bit((code>>j)&1);
    }
    return py::bytes(out);
}

py::bytes decode(const py::bytes& packet_object,size_t count) {
    std::string packet=packet_object; size_t pos=0;
    if(count>packet.size()*8) throw std::invalid_argument("impossible bypass count");
    std::string out; out.reserve(count*2);
    auto bit=[&]() -> unsigned {
        if(pos>=packet.size()*8) throw std::invalid_argument("truncated Golomb bypass");
        unsigned value=(static_cast<unsigned char>(packet[pos/8])>>(7-pos%8))&1; ++pos; return value;
    };
    for(size_t i=0;i<count;++i) {
        unsigned k=0;
        while(!bit()) if(++k>15) throw std::invalid_argument("oversized Golomb value");
        unsigned code=1;
        for(unsigned j=0;j<k;++j) code=(code<<1)|bit();
        unsigned u=code-1, mag=u/2+4;
        if(mag>32768 || (mag==32768 && !(u&1))) throw std::invalid_argument("int16 bypass overflow");
        int q=(u&1) ? -static_cast<int>(mag) : static_cast<int>(mag);
        uint16_t value=static_cast<uint16_t>(q);
        out.push_back(static_cast<char>(value&255));out.push_back(static_cast<char>(value>>8));
    }
    if(packet.size()*8-pos>7) throw std::invalid_argument("unused Golomb bytes");
    while(pos<packet.size()*8) if(bit()) throw std::invalid_argument("nonzero Golomb padding");
    return py::bytes(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("encode",&encode);m.def("decode",&decode);
}
