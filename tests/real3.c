
/*************************************************
*        Join a file onto the output string      *
*************************************************/

/* This is used for readfile/readsock and after a run expansion.
It joins the contents of a file onto the output string, globally replacing
newlines with a given string (optionally).

Arguments:
  f            the FILE
  yield        pointer to the expandable string struct
  eol          newline replacement string, or NULL

Returns:       new pointer for expandable string, terminated if non-null
*/

gstring 
cat_file(FILE * f, gstring * yield, uschar * eol)
{
uschar buffer[1024];

while (Ufgets(buffer, sizeof(buffer), f))
  {
  int len = Ustrlen(buffer);
  if (eol && buffer[len-1] == '\n') len--;
  yield = string_catn(yield, buffer, len);
  if (eol && buffer[len])
    yield = string_cat(yield, eol);
  }
return yield;
}
