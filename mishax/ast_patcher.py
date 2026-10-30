# coding=utf-8
# Copyright 2024 DeepMind Technologies Limited.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AST patcher."""
import ast
import builtins
from collections.abc import Callable, Iterator, Mapping, Sequence
import contextlib
import copy
import dataclasses
import importlib
import inspect
import itertools
import os
import sys
import tempfile
import textwrap
import types
from typing import ContextManager, TypeVar, cast
from unittest import mock
import weakref

import immutabledict



@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PatchSettings:
  """Settings for ModuleASTPatcher.

  Attributes:
    prefix: A section of code to add near the top of the patched module, before
      the patched module members themselves.
    allow_num_matches_upto: The maximum number of places to apply each patch;
      specified as a mapping from member names. If this is not specified, or if
      no value is provided for a particular module member, the patcher will fail
      if there isn't exactly one match for each patch.
  """
  prefix: str | None = None
  allow_num_matches_upto: Mapping[str, int] | None = None


class PatchError(ValueError):
  """Raised when patches don't match the target source."""


_ASTNodeT = TypeVar('_ASTNodeT', bound=ast.AST)


@dataclasses.dataclass(frozen=True, slots=True)
class _CallCapture:
  args: list[ast.expr]
  target_keywords: list[ast.keyword]
  matched_kwarg_names: set[str]
  num_matched_splats: int


_Capture = ast.expr | list[ast.stmt] | list[ast.expr] | _CallCapture


def _ast_undump(dumped_ast: str) -> ast.AST:
  """Inverse of ast.dump."""
  return eval(dumped_ast, vars(ast) | vars(builtins))  # pylint: disable=eval-used


def _ast_defines_symbol(node: ast.AST, symbol_name: str) -> bool:
  """Returns True if the AST node defines the symbol `symbol_name`."""
  if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
    return node.name == symbol_name
  targets = ()
  if isinstance(node, ast.Assign):
    targets = node.targets
  elif isinstance(node, ast.AnnAssign):
    targets = [node.target]
  for n in itertools.chain.from_iterable(ast.walk(t) for t in targets):
    if (
        isinstance(n, ast.Name)
        and n.id == symbol_name
        and isinstance(n.ctx, ast.Store)
    ):
      return True
  return False


def _is_ellipsis_expr(node: ast.AST) -> bool:
  return isinstance(node, ast.Constant) and node.value is Ellipsis


def _is_ellipsis_stmt(node: ast.AST) -> bool:
  return isinstance(node, ast.Expr) and _is_ellipsis_expr(node.value)


def _iter_wildcard_candidate_nodes(node: ast.AST) -> Iterator[ast.AST]:
  """Yields AST nodes excluding contexts where `...` is literal."""
  yield node
  for field, val in ast.iter_fields(node):
    if isinstance(node, ast.Subscript) and field == 'slice':
      continue
    if isinstance(val, list):
      for item in val:
        if isinstance(item, ast.AST):
          yield from _iter_wildcard_candidate_nodes(item)
    elif isinstance(val, ast.AST):
      yield from _iter_wildcard_candidate_nodes(val)


def _has_ellipsis(node: ast.AST) -> bool:
  return any(_is_ellipsis_expr(n) for n in _iter_wildcard_candidate_nodes(node))


def _match_seq_with_ellipsis(
    pattern_seq: Sequence[_ASTNodeT],
    target_seq: Sequence[_ASTNodeT],
    is_ellipsis: Callable[[_ASTNodeT], bool],
    captures: list[_Capture],
    make_capture: Callable[[list[_ASTNodeT]], _Capture],
) -> bool:
  """Matches `pattern_seq` against `target_seq` allowing at most one `...` slice."""
  indices = [i for i, x in enumerate(pattern_seq) if is_ellipsis(x)]
  if len(indices) > 1:
    raise PatchError(
        'At most one `...` is allowed per sequence in patch patterns.'
    )
  if not indices:
    return len(pattern_seq) == len(target_seq) and all(
        _match_ast_with_ellipsis(pat_item, tgt_item, captures)
        for pat_item, tgt_item in zip(pattern_seq, target_seq)
    )
  idx = indices[0]
  prefix, suffix = pattern_seq[:idx], pattern_seq[idx + 1 :]
  if len(target_seq) < len(prefix) + len(suffix):
    return False
  mid_end = len(target_seq) - len(suffix)
  for pat_item, tgt_item in zip(prefix, target_seq[: len(prefix)]):
    if not _match_ast_with_ellipsis(pat_item, tgt_item, captures):
      return False
  matched_slice = copy.deepcopy(list(target_seq[len(prefix) : mid_end]))
  captures.append(make_capture(matched_slice))
  for pat_item, tgt_item in zip(suffix, target_seq[mid_end:]):
    if not _match_ast_with_ellipsis(pat_item, tgt_item, captures):
      return False
  return True


def _match_call_keywords(
    pattern_keywords: Sequence[ast.keyword],
    target_keywords: Sequence[ast.keyword],
    captures: list[_Capture],
) -> bool:
  """Matches `pattern_keywords` as a subset of `target_keywords`."""
  target_kw_by_name = {
      kw.arg: kw for kw in target_keywords if kw.arg is not None
  }
  target_kw_splats = [kw for kw in target_keywords if kw.arg is None]
  splat_idx = 0
  for pat_kw in pattern_keywords:
    if pat_kw.arg is None:
      if splat_idx >= len(target_kw_splats):
        return False
      if not _match_ast_with_ellipsis(
          pat_kw.value, target_kw_splats[splat_idx].value, captures
      ):
        return False
      splat_idx += 1
    else:
      if pat_kw.arg not in target_kw_by_name:
        return False
      if not _match_ast_with_ellipsis(
          pat_kw.value, target_kw_by_name[pat_kw.arg].value, captures
      ):
        return False
  return True


def _match_ast_with_ellipsis(
    pattern: ast.AST, target: ast.AST, captures: list[_Capture]
) -> bool:
  """Recursively matches pattern AST `pattern` against `target`, recording `...` bindings."""
  if _is_ellipsis_expr(pattern):
    if not isinstance(target, ast.expr):
      return False
    captures.append(copy.deepcopy(target))
    return True
  if type(pattern) is not type(target):
    return False

  if isinstance(pattern, ast.Call) and isinstance(target, ast.Call):
    if any(_is_ellipsis_expr(a) for a in pattern.args):
      if not _match_ast_with_ellipsis(pattern.func, target.func, captures):
        return False
      matched_kwarg_names = {
          kw.arg for kw in pattern.keywords if kw.arg is not None
      }
      num_matched_splats = sum(1 for kw in pattern.keywords if kw.arg is None)
      if not _match_seq_with_ellipsis(
          pattern.args,
          target.args,
          _is_ellipsis_expr,
          captures,
          lambda matched_args: _CallCapture(
              args=matched_args,
              target_keywords=copy.deepcopy(list(target.keywords)),
              matched_kwarg_names=matched_kwarg_names,
              num_matched_splats=num_matched_splats,
          ),
      ):
        return False
      return _match_call_keywords(pattern.keywords, target.keywords, captures)

  if isinstance(pattern, (ast.Tuple, ast.List)) and isinstance(
      target, (ast.Tuple, ast.List)
  ):
    return type(pattern.ctx) is type(target.ctx) and _match_seq_with_ellipsis(
        pattern.elts, target.elts, _is_ellipsis_expr, captures, list
    )

  for field, pat_val in ast.iter_fields(pattern):
    target_val = getattr(target, field, None)
    if isinstance(pattern, ast.Subscript) and field == 'slice':
      if not isinstance(pat_val, ast.AST) or not isinstance(
          target_val, ast.AST
      ):
        return False
      if ast.dump(pat_val) != ast.dump(target_val):
        return False
      continue
    if isinstance(pat_val, list) and isinstance(target_val, list):
      if field in ('body', 'orelse', 'finalbody') and all(
          isinstance(x, ast.stmt) for x in pat_val
      ):
        pat_stmts = cast(list[ast.stmt], pat_val)
        tgt_stmts = cast(list[ast.stmt], target_val)
        if not _match_seq_with_ellipsis(
            pat_stmts, tgt_stmts, _is_ellipsis_stmt, captures, list
        ):
          return False
      else:
        if len(pat_val) != len(target_val):
          return False
        for pat_item, tgt_item in zip(pat_val, target_val):
          if isinstance(pat_item, ast.AST) and isinstance(tgt_item, ast.AST):
            if not _match_ast_with_ellipsis(pat_item, tgt_item, captures):
              return False
          elif pat_item != tgt_item:
            return False
    elif isinstance(pat_val, ast.AST) and isinstance(target_val, ast.AST):
      if not _match_ast_with_ellipsis(pat_val, target_val, captures):
        return False
    elif pat_val != target_val:
      return False
  return True


def _next_capture(captures_iter: Iterator[_Capture]) -> _Capture:
  try:
    return next(captures_iter)
  except StopIteration as e:
    raise PatchError(
        'Replacement pattern has more `...` wildcards than match pattern.'
    ) from e


def _instantiate_seq_with_ellipsis(
    seq: Sequence[_ASTNodeT],
    is_ellipsis: Callable[[_ASTNodeT], bool],
    captures_iter: Iterator[_Capture],
    expand_capture: Callable[[_Capture], Sequence[_ASTNodeT]],
) -> list[_ASTNodeT]:
  """Instantiates `seq`, expanding at most one `...` element via `expand_capture`."""
  if sum(1 for x in seq if is_ellipsis(x)) > 1:
    raise PatchError(
        'At most one `...` is allowed per sequence in patch patterns.'
    )
  out: list[_ASTNodeT] = []
  for item in seq:
    if is_ellipsis(item):
      out.extend(
          copy.deepcopy(list(expand_capture(_next_capture(captures_iter))))
      )
    else:
      out.append(_instantiate_ast_with_ellipsis(item, captures_iter))
  return out


def _merge_call_keywords(
    call_capture: _CallCapture,
    after_keywords: Sequence[ast.keyword],
) -> list[ast.keyword]:
  """3-way merges captured call keywords with `after_keywords`."""
  after_kw_by_name = {kw.arg: kw for kw in after_keywords if kw.arg is not None}
  after_kw_splats = [kw for kw in after_keywords if kw.arg is None]
  merged: list[ast.keyword] = []
  splat_idx = 0
  for target_kw in call_capture.target_keywords:
    if target_kw.arg is None:
      if splat_idx < call_capture.num_matched_splats:
        if splat_idx < len(after_kw_splats):
          merged.append(after_kw_splats[splat_idx])
        splat_idx += 1
      else:
        merged.append(copy.deepcopy(target_kw))
    elif target_kw.arg in after_kw_by_name:
      merged.append(after_kw_by_name.pop(target_kw.arg))
    elif target_kw.arg not in call_capture.matched_kwarg_names:
      merged.append(copy.deepcopy(target_kw))
  merged.extend(after_kw_by_name.values())
  merged.extend(after_kw_splats[splat_idx:])
  return merged


def _instantiate_ast_with_ellipsis(
    after: _ASTNodeT, captures_iter: Iterator[_Capture]
) -> _ASTNodeT:
  """Recursively substitutes `...` bindings from `captures_iter` into `after`."""
  if _is_ellipsis_expr(after):
    capture = _next_capture(captures_iter)
    if not isinstance(capture, ast.expr):
      raise PatchError(
          'Expected expression capture for `...`, got'
          f' {type(capture).__name__}.'
      )
    return cast(_ASTNodeT, copy.deepcopy(capture))

  if isinstance(after, ast.Call) and any(
      _is_ellipsis_expr(arg) for arg in after.args
  ):
    call_capture: _CallCapture | None = None

    def _expand_call_arg_capture(capture: _Capture) -> Sequence[ast.expr]:
      nonlocal call_capture
      if isinstance(capture, _CallCapture):
        call_capture = capture
        return capture.args
      if isinstance(capture, ast.expr):
        return [capture]
      raise PatchError(
          f'Cannot splice {type(capture).__name__} into call arguments.'
      )

    new_func = _instantiate_ast_with_ellipsis(after.func, captures_iter)
    new_args = _instantiate_seq_with_ellipsis(
        after.args, _is_ellipsis_expr, captures_iter, _expand_call_arg_capture
    )
    after_keywords = [
        _instantiate_ast_with_ellipsis(kw, captures_iter)
        for kw in after.keywords
    ]
    new_keywords = (
        _merge_call_keywords(call_capture, after_keywords)
        if call_capture is not None
        else after_keywords
    )
    return cast(
        _ASTNodeT, ast.Call(func=new_func, args=new_args, keywords=new_keywords)
    )

  if isinstance(after, (ast.Tuple, ast.List)):

    def _expand_elts(capture: _Capture) -> Sequence[ast.expr]:
      if isinstance(capture, list):
        return cast(list[ast.expr], capture)
      if isinstance(capture, ast.expr):
        return [capture]
      raise PatchError(
          f'Cannot splice {type(capture).__name__} into tuple/list elements.'
      )

    res = copy.copy(after)
    res.elts = _instantiate_seq_with_ellipsis(
        after.elts, _is_ellipsis_expr, captures_iter, _expand_elts
    )
    res.ctx = copy.deepcopy(after.ctx)
    return res

  res = copy.copy(after)
  for field, field_val in ast.iter_fields(after):
    if isinstance(after, ast.Subscript) and field == 'slice':
      setattr(res, field, copy.deepcopy(field_val))
      continue
    if isinstance(field_val, list):
      if field in ('body', 'orelse', 'finalbody') and all(
          isinstance(x, ast.stmt) for x in field_val
      ):

        def _expand_stmts(capture: _Capture) -> Sequence[ast.stmt]:
          if not isinstance(capture, list):
            raise PatchError(
                f'Cannot splice {type(capture).__name__} into statement block.'
            )
          return cast(list[ast.stmt], capture)

        setattr(
            res,
            field,
            _instantiate_seq_with_ellipsis(
                field_val, _is_ellipsis_stmt, captures_iter, _expand_stmts
            ),
        )
      else:
        setattr(
            res,
            field,
            [
                _instantiate_ast_with_ellipsis(x, captures_iter)
                if isinstance(x, ast.AST)
                else copy.deepcopy(x)
                for x in field_val
            ],
        )
    elif isinstance(field_val, ast.AST):
      setattr(
          res, field, _instantiate_ast_with_ellipsis(field_val, captures_iter)
      )
    else:
      setattr(res, field, copy.deepcopy(field_val))
  return res


def _instantiate_target(
    after_node: _ASTNodeT,
    captures: Sequence[_Capture],
) -> _ASTNodeT:
  """Instantiates `after_node` using `captures`, raising `PatchError` on mismatch."""
  captures_iter = iter(captures)
  res = _instantiate_ast_with_ellipsis(after_node, captures_iter)
  if next(captures_iter, None) is not None:
    raise PatchError(
        'Match pattern has more `...` wildcards than replacement pattern.'
    )
  return res


def _apply_ellipsis_patch(
    before: ast.Module,
    after: ast.Module,
    is_expr_patch: bool,
    dumped_ast: str,
) -> tuple[str, int]:
  """Applies a patch containing `...` to `dumped_ast`, returning `(new_dumped_ast, num_matches)`."""
  if any(_is_ellipsis_stmt(s) for s in before.body):
    raise PatchError(
        'Top-level `...` statements are not supported in patch patterns.'
    )
  tree = _ast_undump(dumped_ast)
  num_matches = 0

  if is_expr_patch:
    before_expr = cast(ast.Expr, before.body[0]).value
    after_expr = cast(ast.Expr, after.body[0]).value

    class _ExprReplacer(ast.NodeTransformer):

      def generic_visit(self, node: ast.AST) -> ast.AST:
        nonlocal num_matches
        if isinstance(node, ast.expr):
          captures: list[_Capture] = []
          if _match_ast_with_ellipsis(before_expr, node, captures):
            num_matches += 1
            return _instantiate_target(after_expr, captures)
        return super().generic_visit(node)

    return ast.dump(_ExprReplacer().visit(tree)), num_matches

  num_pattern_stmts = len(before.body)
  if num_pattern_stmts == 0:
    return dumped_ast, 0

  def _replace_stmt_list(stmts: list[ast.stmt]) -> list[ast.stmt]:
    nonlocal num_matches
    new_stmts: list[ast.stmt] = []
    i = 0
    while i < len(stmts):
      if i <= len(stmts) - num_pattern_stmts:
        target_slice = stmts[i : i + num_pattern_stmts]
        captures: list[_Capture] = []
        if _match_seq_with_ellipsis(
            before.body, target_slice, _is_ellipsis_stmt, captures, list
        ):
          captures = [
              _replace_stmt_list(cast(list[ast.stmt], c))
              if isinstance(c, list) and all(isinstance(s, ast.stmt) for s in c)
              else c
              for c in captures
          ]
          num_matches += 1
          new_stmts.extend(_instantiate_target(after, captures).body)
          i += num_pattern_stmts
          continue
      _replace_stmts_in_node(stmts[i])
      new_stmts.append(stmts[i])
      i += 1
    return new_stmts

  def _replace_stmts_in_node(node: ast.AST) -> None:
    for field, val in ast.iter_fields(node):
      if (
          field in ('body', 'orelse', 'finalbody')
          and isinstance(val, list)
          and all(isinstance(s, ast.stmt) for s in val)
      ):
        setattr(node, field, _replace_stmt_list(val))
      elif isinstance(val, list):
        for item in val:
          if isinstance(item, ast.AST):
            _replace_stmts_in_node(item)
      elif isinstance(val, ast.AST):
        _replace_stmts_in_node(val)

  _replace_stmts_in_node(tree)
  return ast.dump(tree), num_matches


_INSTALLED_PATCHER_CONTEXTS = dict['ModuleASTPatcher', ContextManager[None]]()


class ModuleASTPatcher(Callable[[], ContextManager[None]]):  # pyrefly: ignore[invalid-inheritance]
  """Creates a patcher that applies a series of patches to a module.

  The patched module members are in a separate temporary source file, visible to
  stacktraces and debuggers.

  `prefix` is code that should be run at the top of the file.

  For each object, the patches are logically a sequence of pairs (before, after)
  to match and replace; the patcher will error if there isn't at least one match
  (exactly one, unless more are specified via `settings`).
  For formatting convenience, the patcher will also accept a list of source
  blocks in alternating (before1, after1, before2, after2) order.

  The patcher may then be used as a `with patcher():` context manager, or using
  `patcher.install()` or `patcher.install_clean()`. These are in increasing
  order of robustness, and decreasing order of flexibility:
    * the `patcher()` context manager is flexible and useful for prototyping or
      debugging, but it changes and unchanges the target module, which isn't
      threadsafe.
    * `install_clean()` applies the patches on the target module as it's
      imported, which ensures consistency but requires the module to be
      specified as a string and not already imported.
    * `install()` is between the two. It's threadsafe, and can be called at any
      time by downstream code, but may occasionally produce incorrect results
      due to aliasing by other modules.

  NOTE: Patching a module on the source level may produce surprises additional
  to those implicitly mentioned above:
  1. If the target module is changed (e.g. via a version upgrade) then the patch
  could fail to apply, or (worse) apply but with changed meaning depending on
  the underlying code changes. Normally code maintainers will give consideration
  to downstream code by being careful with changes to the interface (e.g., type
  signatures), but will assume significant freedom in changing the
  implementation if the resulting behaviour is unchanged. However, this will
  still break patches, which depend on the implementation rather than the
  interface -- so even careful maintainers are under no obligation not to break
  patches.
  2. If the target module member definitions change global state (e.g. by
  registering themselves somewhere) then that state change will happen again
  when the patch is applied.
  """

  def __init__(
      self,
      module: types.ModuleType | str,
      settings: PatchSettings = PatchSettings(),
      /,
      **patches_per_object: Sequence[str] | Sequence[tuple[str, str]],
  ):
    self.module = module
    self._settings = settings
    self._patches_per_object = patches_per_object
    self._updated_members: immutabledict.immutabledict[str, object] | None = (
        None
    )
    self._original_members: immutabledict.immutabledict[str, object] | None = (
        None
    )
    self._src: str | None = None
    self._path: str | None = None
    self._globals: dict[str, object] | None = None
    if not isinstance(self.module, str):
      self._updated_members = self.updated_members

  def __str__(self) -> str:
    return f'ModuleASTPatcher({self.module_name})'

  def install(self) -> None:
    """Installs the patches on the target module.

    This is a cleaner way of using the patcher than via `__call__`, because it
    ensures code will subsequently always see the patched version of the module.

    It's less clean than `install_clean`; in exchange it allows the install to
    take place at some convenient time rather than necessarily having to precede
    the import of the patched module.

    However, transitioning between these different modes should be smooth -- you
    may .install_clean(), then later .install(), and enter a __call__() context,
    and the earlier things in this chain will turn the later ones into no-ops
    (if using the same ModuleASTPatcher instance).
    """
    if self not in _INSTALLED_PATCHER_CONTEXTS:
      context = self()
      context.__enter__()
      _INSTALLED_PATCHER_CONTEXTS[self] = context

  def is_installed(self) -> bool:
    """Returns whether the patcher is currently installed."""
    return self in _INSTALLED_PATCHER_CONTEXTS

  def install_clean(self) -> None:
    """Installs the patches on top of the freshly-imported module.

    This requires the module to be specified as a string, and will raise an
    error if the module is already imported. The benefit is that this ensures
    any other module that imports the patched module will get the patched
    version, rather than e.g. aliasing some unpatched module members, leading to
    subtle bugs.

    Usage:
      patcher = ModuleASTPatcher("some.fantastic.module", ...)
      patcher.install_clean()
    """
    if not isinstance(self.module, str) or self.module in sys.modules:
      raise ValueError(f'Module {self.module} already imported.')
    self.install()

  @contextlib.contextmanager
  def __call__(self):
    with contextlib.ExitStack() as stack:
      if self not in _INSTALLED_PATCHER_CONTEXTS:
        for name, value in self.updated_members.items():
          stack.enter_context(mock.patch.object(self.module, name, value))
      yield

  @property
  def module_name(self) -> str:
    return self.module if isinstance(self.module, str) else self.module.__name__

  @property
  def updated_members(self) -> immutabledict.immutabledict[str, object]:
    if self._updated_members is None:
      self._setup()
    assert self._updated_members is not None
    return self._updated_members

  @property
  def original_members(self) -> immutabledict.immutabledict[str, object]:
    if self._original_members is None:
      self._setup()
    assert self._original_members is not None
    return self._original_members

  @property
  def globals(self) -> dict[str, object]:
    if self._globals is None:
      self._setup()
    assert self._globals is not None
    return self._globals

  @property
  def src(self) -> str:
    if self._src is None:
      self._setup()
    assert self._src is not None
    return self._src

  @property
  def path(self) -> str:
    if self._path is None:
      self._setup()
    assert self._path is not None
    return self._path

  def _setup(self):
    """Sets up the patcher source and temporary file."""
    module_name = self.module_name
    if isinstance(self.module, str):
      self.module = importlib.import_module(module_name)
    self._original_members = immutabledict.immutabledict(
        {name: getattr(self.module, name) for name in self._patches_per_object}
    )
    src_blocks = [
        'import sys',
        (
            '# Import existing members of target module into patcher module.\n'
            f'globals().update(sys.modules[{module_name!r}].__dict__)'
        ),
    ]
    allow_num_matches_upto = self._settings.allow_num_matches_upto or {}
    if self._settings.prefix is not None:
      src_blocks.append(self._settings.prefix)
    for name in set(allow_num_matches_upto) - set(self._patches_per_object):
      raise ValueError(
          f'Unpatched module member {name} in settings.allow_num_matches_upto'
      )
    for name, patches in self._patches_per_object.items():
      target_src = inspect.getsource(getattr(self.module, name))
      try:
        parsed_target = ast.parse(target_src)
      except SyntaxError as e:
        raise PatchError(
            f'Cannot patch {module_name}.{name}: inspect.getsource() returned'
            ' invalid syntax. The target member may have already been'
            f' monkey-patched:\n```\n{target_src}\n```'
        ) from e
      if not any(
          _ast_defines_symbol(node, name) for node in parsed_target.body
      ):
        raise PatchError(
            f'Cannot patch {module_name}.{name}: source retrieved by'
            f" inspect.getsource() does not define '{name}'. The target member"
            ' may have already been monkey-patched. Returned'
            f' snippet:\n```\n{target_src}\n```'
        )
      dumped_ast = ast.dump(parsed_target)
      iter_patches = iter(patches)
      for patch in iter_patches:
        if isinstance(patch, str):
          # This is to support the convenient "flat" input format where multiple
          # patches can be provided as (before1, after1, before2, after2, ...).
          before_raw, after_raw = patch, next(iter_patches)
          assert isinstance(after_raw, str)
        else:
          before_raw, after_raw = patch
        before_src = textwrap.dedent(before_raw).strip()
        after_src = textwrap.dedent(after_raw).strip()
        before, after = ast.parse(before_src), ast.parse(after_src)

        match before.body:
          case [ast.Expr(before_expr_value)]:
            pass
          case _:
            before_expr_value = None
        is_expr_patch = False
        match before_expr_value, after.body:
          case ast.expr(), [ast.Expr(after_expr_value)]:
            # Special case for patches that are expressions (rather than one or
            # more statements).
            is_expr_patch = True
            before_dumped_ast = ast.dump(before_expr_value)
            after_dumped_ast = ast.dump(after_expr_value)
          case _:
            # In an AST dump with multiple statements, the statements will be
            # separated by ', '. An alternative is to ast.dump the before and
            # after nodes themselves, but that comes with more unwanted wrapping
            # text.
            before_dumped_ast = ', '.join(map(ast.dump, before.body))
            after_dumped_ast = ', '.join(map(ast.dump, after.body))
        before_has_ellipsis = _has_ellipsis(before)
        if before_has_ellipsis:
          dumped_ast, num_matches = _apply_ellipsis_patch(
              before, after, is_expr_patch, dumped_ast
          )
        elif _has_ellipsis(after):
          raise PatchError(
              'Replacement pattern contains `...` but match pattern does not.'
          )
        else:
          num_matches = dumped_ast.count(before_dumped_ast)
          dumped_ast = dumped_ast.replace(before_dumped_ast, after_dumped_ast)
        if not num_matches:
          expr_in_ast = False
          if not is_expr_patch and before_expr_value is not None:
            if before_has_ellipsis:
              expr_in_ast = any(
                  _match_ast_with_ellipsis(before_expr_value, n, [])
                  for n in ast.walk(_ast_undump(dumped_ast))
                  if isinstance(n, ast.expr)
              )
            else:
              expr_in_ast = ast.dump(before_expr_value) in dumped_ast
          if expr_in_ast:
            raise PatchError(
                f'The AST of {module_name}.{name} contains the AST of'
                f' ```\n{before_src}\n``` interpreted as an expression but not'
                ' as statement(s). This is an error because the replacement'
                ' code in the patch is statement(s), not an expression.'
            )
          else:
            raise PatchError(
                f'No match in the AST of {module_name}.{name} for the AST of'
                f' ```\n{before_src}\n```. The target source is:'
                f' ```\n{target_src}\n```.'
            )
        if num_matches > allow_num_matches_upto.get(name, 1):
          raise PatchError(
              f'Too many ({num_matches}) matches in the AST of'
              f' {module_name}.{name} for the AST of ```\n{before_src}\n```.'
              f' The target source is: ```\n{target_src}\n```. Consider'
              ' adjusting the PatchSettings.allow_num_matches_upto field.'
          )
      patched_ast = ast.fix_missing_locations(_ast_undump(dumped_ast))
      src_blocks.append(ast.unparse(patched_ast))
    self._src = '\n\n'.join(src_blocks)
    # Create a file for stacktraces and debuggers to find source for patched
    # objects.
    fd, self._path = tempfile.mkstemp(
        prefix=f'{module_name}.', suffix='.PATCHED.py', text=True
    )
    weakref.finalize(self, os.remove, self._path)
    os.write(fd, self._src.encode())
    os.close(fd)
    exec_globals: dict[str, object] = {}
    exec(compile(self._src, self._path, 'exec'), exec_globals)  # pylint: disable=exec-used
    self._globals = exec_globals
    updated_members = {
        name: exec_globals[name] for name in self._patches_per_object
    }
    self._updated_members = immutabledict.immutabledict(updated_members)
