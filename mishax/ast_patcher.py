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
from collections.abc import Callable, Mapping, Sequence
import contextlib
import dataclasses
import inspect
import os
import tempfile
import types
from typing import ContextManager
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


def _ast_undump(dumped_ast: str) -> ast.AST:
  """Inverse of ast.dump."""
  return eval(dumped_ast, vars(ast) | vars(builtins))  # pylint: disable=eval-used


class ModuleASTPatcher(Callable[[], ContextManager[None]]):
  """Creates a patcher that applies a series of patches to a module.

  The patched module members are in a separate temporary source file, visible to
  stacktraces and debuggers.

  `prefix` is code that should be run at the top of the file.

  For each object, the patches are logically a sequence of pairs (before, after)
  to match and replace; the patcher will error if there isn't at least one match
  (exactly one, unless more are specified via `settings`).
  For formatting convenience, the patcher will also accept a list of source
  blocks in alternating (before1, after1, before2, after2) order.

  `patcher()` is then a context manager, to be used in a `with` statement, or
  applied indefinitely with `patcher().__enter__().
  """

  def __init__(
      self,
      module: types.ModuleType,
      settings: PatchSettings = PatchSettings(),
      /,
      **patches_per_object: str | Sequence[str] | Sequence[tuple[str, str]],
  ):
    src_blocks = [
        'import sys',
        (
            '# Import existing members of target module into patcher module.\n'
            f'globals().update(sys.modules[{module.__name__!r}].__dict__)'
        ),
    ]
    allow_num_matches_upto = settings.allow_num_matches_upto or {}
    if settings.prefix is not None:
      src_blocks.append(settings.prefix)
    for name in set(allow_num_matches_upto) - set(patches_per_object):
      raise ValueError(
          f'Unpatched module member {name} in settings.allow_num_matches_upto'
      )
    for name, patches in patches_per_object.items():
      dumped_ast = ast.dump(ast.parse(inspect.getsource(getattr(module, name))))
      iter_patches = iter(patches)
      for patch in iter_patches:
        if isinstance(patch, str):
          # This is to support the convenient "flat" input format where multiple
          # patches can be provided as (before1, after1, before2, after2, ...).
          before, after = patch, next(iter_patches)
        else:
          before, after = patch
        before, after = ast.parse(before.strip()), ast.parse(after.strip())

        match before.body, after.body:
          case [ast.Expr(before_value)], [ast.Expr(after_value)]:
            # Special case for patches that are expressions (rather than one or
            # more statements).
            before_dumped_ast = ast.dump(before_value)
            after_dumped_ast = ast.dump(after_value)
          case _:
            # In an AST dump with multiple statements, the statements will be
            # separated by ', '. An alternative is to ast.dump the before and
            # after nodes themselves, but that comes with more unwanted wrapping
            # text.
            before_dumped_ast = ', '.join(map(ast.dump, before.body))
            after_dumped_ast = ', '.join(map(ast.dump, after.body))
        num_matches = dumped_ast.count(before_dumped_ast)
        if not num_matches:
          raise ValueError(
              f'No match for {before_dumped_ast} in'
              f' {ast.dump(_ast_undump(dumped_ast), indent=2)}'
          )
        if num_matches > allow_num_matches_upto.get(name, 1):
          raise ValueError(
              f'Too many matches for {before_dumped_ast} in'
              f' {ast.dump(_ast_undump(dumped_ast), indent=2)}'
          )
        dumped_ast = dumped_ast.replace(before_dumped_ast, after_dumped_ast)
      patched_ast = ast.fix_missing_locations(_ast_undump(dumped_ast))
      src_blocks.append(ast.unparse(patched_ast))
    self.module = module
    self.src = '\n\n'.join(src_blocks)
    # Create a file for stacktraces and debuggers to find source for patched
    # objects.
    fd, self.path = tempfile.mkstemp(suffix='.py', text=True)
    weakref.finalize(self, os.remove, self.path)
    os.write(fd, self.src.encode())
    os.close(fd)
    exec(compile(self.src, self.path, 'exec'), envt := {})  # pylint: disable=exec-used
    updated_members = {name: envt[name] for name in patches_per_object}
    self.updated_members = immutabledict.immutabledict(updated_members)

  @contextlib.contextmanager
  def __call__(self):
    with contextlib.ExitStack() as stack:
      for name, value in self.updated_members.items():
        stack.enter_context(mock.patch.object(self.module, name, value))
      yield
