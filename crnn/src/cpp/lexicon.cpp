#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <algorithm>

#include <lua.hpp>
#include <luaT.h>
#include <thpp/Tensor.h>
#include <thpp/Storage.h>
#include <fblualib/LuaUtils.h>

#include "bk_tree.h"
#include "utils.hpp"

using namespace std;

template <class T>
int createLexiconFromFile(lua_State* L) {
    // arg1: lexicon file path
    luaL_checkstring(L, 1);
    string lexiconFpath = string(lua_tostring(L, 1));

    auto* bktree = new bk::BKTree<string, int>();
    ifstream lexFile(lexiconFpath);
    string tline;
    while (std::getline(lexFile, tline)) {
        bktree->insert(tline);
    }

    lua_pushlightuserdata(L, (void*)bktree);
    return 1;
}

std::vector<std::string> getListOfStrings(lua_State* L, int idx) {
    luaL_checktype(L, idx, LUA_TTABLE);
    std::vector<std::string> strings;
    int n = lua_objlen(L, idx);
    for (int i = 0; i < n; ++i) {
        lua_pushinteger(L, i+1);
        lua_gettable(L, idx);
        const char* cstr = lua_tostring(L, -1);
        strings.push_back(std::string(cstr));
        lua_pop(L, 1);
    }
    return strings;
}

template <class T>
int createLexiconFromStringList(lua_State* L) {
    // luaL_checkstring(L, 1);
    // string lexiconFpath = string(lua_tostring(L, 1));
    auto strings = getListOfStrings(L, 1);

    auto* bktree = new bk::BKTree<string, int>();
    for (auto str : strings) {
        bktree->insert(str);
    }

    lua_pushlightuserdata(L, (void*)bktree);
    return 1;
}

template <class T>
int destroyLexicon(lua_State* L) {
    auto* bktree = (bk::BKTree<string, int>*)lua_touserdata(L, 1);
    if (bktree != NULL) {
        delete bktree;
    }
    return 0;
}

template <class T>
int searchWithin(lua_State* L) {
    // arg1: lex pointer
    auto* bktree = (bk::BKTree<string, int>*)lua_touserdata(L, 1);
    // arg2: query
    const char* cstr = lua_tostring(L, 2);
    string queryStr(cstr);
    // arg3: range
    int searchRange = lua_tonumber(L, 3);
    // arg4: maxK
    int maxK = lua_tonumber(L, 4);

    auto searchResults = bktree->find_within(queryStr, searchRange);
    std::stable_sort(searchResults.begin(), searchResults.end(),
        [](const std::pair<string, int>& a, const std::pair<string, int>& b) {
            return a.second <= b.second;
        });
    int nReturn = std::min(maxK, (int)searchResults.size());
    // result strings
    lua_newtable(L);
    for (int i = 0; i < nReturn; ++i) {
        lua_pushnumber(L, i+1);
        lua_pushstring(L, searchResults[i].first.c_str());
        lua_settable(L, -3);
    }
    // result distances
    lua_newtable(L);
    for (int i = 0; i < nReturn; ++i) {
        lua_pushnumber(L, i+1);
        lua_pushnumber(L, searchResults[i].second);
        lua_settable(L, -3);
    }

    return 2;
}

// register functions
template <class T>
class Registerer_lex {
private:
    static const luaL_Reg functions_[];
public:
    static void registerFunctions(lua_State* L);
};

template <class T>
const luaL_Reg Registerer_lex<T>::functions_[] = {
    {"LEX_createLexiconFromFile",       createLexiconFromFile<T>},
    {"LEX_createLexiconFromStringList", createLexiconFromStringList<T>},
    {"LEX_destroyLexicon",              destroyLexicon<T>},
    {"LEX_searchWithin",                searchWithin<T>},
    {nullptr, nullptr},
};

template <class T>
void Registerer_lex<T>::registerFunctions(lua_State* L) {
    luaT_pushmetatable(L, thpp::Tensor<T>::kLuaTypeName);
    luaT_registeratname(L, functions_, "nn");
    lua_pop(L, 1);
}

void initLex(lua_State* L) {
    Registerer_lex<float>::registerFunctions(L);
    Registerer_lex<double>::registerFunctions(L);
}
