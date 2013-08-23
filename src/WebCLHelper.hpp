#ifndef WEBCLVALIDATOR_WEBCLHELPER
#define WEBCLVALIDATOR_WEBCLHELPER

#include "WebCLDebug.hpp"

#include "clang/AST/Type.h"

#include <map>
#include <set>

namespace clang {
    class Expr;
    class ValueDecl;
    class VarDecl;
    class ParmVarDecl;
    class Rewriter;
    class CompilerInstance;
}

/// \brief Mixin class for examining AST nodes.
class WebCLHelper
{
public:

    WebCLHelper();
    ~WebCLHelper();

    /// Assume that the expression refers to a pointer and return the
    /// type of the pointed value.
    clang::QualType getPointeeType(clang::Expr *expr);

    /// Remove implicit casts and parentheses.
    clang::Expr *pruneExpression(clang::Expr *expr);

    /// Prune the expression, assume that it refers to value
    /// declaration and finally return the value declaration.
    clang::ValueDecl *pruneValue(clang::Expr *expr);
};

/// vector is used after when address space variables are ordered and possible
/// paddings are added
typedef std::vector<clang::VarDecl*> AddressSpaceInfo;

/// Contains all information of allocated memory of an address space
///
/// Used to pass information to transformer so that it can write typedefs
/// and initializers for it.
class AddressSpaceLimits {
public:
  AddressSpaceLimits(bool hasStaticallyAllocatedLimits, unsigned addressSpace) :
    hasStaticallyAllocatedLimits_(hasStaticallyAllocatedLimits)
  , addressSpace_(addressSpace){};
  ~AddressSpaceLimits() {};
  
  void insert(clang::ParmVarDecl *parm) {
    dynamicLimits_.push_back(parm);
  };

  bool hasStaticallyAllocatedLimits() { return hasStaticallyAllocatedLimits_; };
  unsigned getAddressSpace() { return addressSpace_; };
  
  bool empty() {
    return !hasStaticallyAllocatedLimits_ && dynamicLimits_.empty();
  };

  unsigned count() {
    return hasStaticallyAllocatedLimits() ?
      (dynamicLimits_.size()+1) : dynamicLimits_.size();
  };
    
  typedef std::vector<clang::ParmVarDecl*> LimitList;
  
  LimitList& getDynamicLimits() { return dynamicLimits_; };

private:
  bool hasStaticallyAllocatedLimits_;
  unsigned addressSpace_;
  LimitList dynamicLimits_;
  
};

/// Improves rewriter so that we can do a little more complex modifications
/// and still query modified rewritten text.
///
/// Doesn't apply replacements directly, but collects them and leaves original
/// SourceLocations untouched.
class WebCLRewriter {
public:
  
  WebCLRewriter(clang::CompilerInstance &instance, clang::Rewriter &rewriter);

  /// \brief Map of modified source ranges and corresponding replacements.
  typedef std::pair< clang::SourceLocation, clang::SourceLocation > WclSourceRange;
  
  typedef std::map<WclSourceRange, std::string> ModificationMap;
  
  /// \brief Looks char by char forward until the position where next
  /// requested character is found from original source.
  clang::SourceLocation findLocForNext(clang::SourceLocation startLoc,
                                       char charToFind);
  
  /// \brief Removes given range.
  void removeText(clang::SourceRange range);
  
  /// \brief Replaces given range.
  void replaceText(clang::SourceRange range, std::string text);

  /// \brief if for asked source range has been added transformations, return
  ///
  /// Returns ransformed result like rewriters getRewrittenText. Combines on the
  /// fly original and modified ranges from requested range.
  ///
  /// e.g. if location is [20, 50] we would find all transformations inside is
  ///      [20,22], [25,30], [33,45] and before that all transformations inside
  ///      those ranges [34,37] etc..
  std::string getTransformedText(clang::SourceRange range);

  /// \brief Get modified source ranges and replacements.
  ///
  /// Returns only toplevel modifications to be easily delegated to rewriter.
  /// each call might regenerate underlying map, so get map once for getting
  /// e.g. begin and end iterators.
  ModificationMap& modifiedRanges();
  
private:
  clang::CompilerInstance &instance_;
  clang::Rewriter &rewriter_;
  
  typedef std::pair< int, int >                  ModifiedRange;
  typedef std::map< ModifiedRange, std::string > RangeModifications;

  // filtered ranges, which has only the top level modifications and does not
  // include nested ones (top level should already contain nested changes as string)
  typedef std::set<ModifiedRange> RangeModificationsFilter;

  /// \brief Returns toplevel modifications to allow applying them to source.
  RangeModificationsFilter& filteredModifiedRanges();
  
  RangeModifications modifiedRanges_;
  RangeModificationsFilter filteredModifiedRanges_;

  /// State if we should regenerate the filtered ranges data
  bool isFilteredRangesDirty_;
  
  /// \brief Used to deliver modification list data outside of this class.
  ModificationMap externalMap_;

};


#endif // WEBCLVALIDATOR_WEBCLHELPER
