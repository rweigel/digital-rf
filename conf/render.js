renderFunctions.crossLink = function (columnName, config) {
  return (columnString, type, row, meta) => {
    if (type !== 'display') {
      return columnString
    }
    const columnNames = config.dataTables.columns.map(c => c.name)
    const tableName = config.dataTablesAdditions.tableMetadata.tableName

    if (tableName === 'samples') {
      const href = `../blocks/#station=${row[0]}&sample=${row[1]}`
      const title = `Blocks for station ${row[0]} sample ${row[1]}`
      columnString = `<a href="${href}" title="${title}" target="_blank">${columnString}</a>`
    }

    if (tableName === 'blocks') {
      const href = `../samples/#station=${row[0]}&sample=${row[1]}`
      const title = `Sample metadata associated with sample ${row[1]}`
      columnString = `<a href="${href}" title="${title}" target="_blank">${columnString}</a>`
    }

    return columnString
  }
}
